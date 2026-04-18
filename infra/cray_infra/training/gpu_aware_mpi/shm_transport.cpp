#include "shm_transport.h"
#include "shm_channel.h"
#include "common.h"

#include <cstring>

static constexpr int SHM_CTRL_TAG = 42;

// Backpressure: sender spins here until the receiver has fully consumed
// enough slots that writing into `write_slot` won't clobber something the
// peer is still reading.  With SHM_NUM_SLOTS=2 the sender may be at most
// one generation ahead of the receiver — writing into slot 0 for
// generation N+1 requires that reader_done has reached N-1 (i.e. slot 0
// from generation N-1 has been fully consumed).
//
// Acquire-load so any writes the peer did to payload / slot metadata before
// incrementing reader_done become visible to us.  The spin is bounded by
// the receiver's copy speed; in practice each generation consumes a few
// hundred microseconds for a 256 MB payload.
static void wait_for_slot(ShmDirState& st) {
    while (true) {
        uint64_t done = st.reader_done_shm->load(std::memory_order_acquire);
        // next_gen is the generation we're ABOUT to stamp on this send.
        // We can only proceed when the receiver has finished through
        // (next_gen - SHM_NUM_SLOTS), i.e. the slot we're about to reuse.
        if (st.next_gen <= done + SHM_NUM_SLOTS) return;
        // Busy-wait: a short stall is typical; yielding to the kernel would
        // only help under severe oversubscription of cores, which isn't the
        // situation for the training job case this extension targets.
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
}

// Receiver-side spin: wait until the slot's seq counter advances to the
// generation the sender's signal promised.  Acquire-load so the payload
// memcpy that preceded the sender's release-store of seq becomes visible.
static void wait_for_gen(ShmSlotHeader* hdr, uint64_t expected_gen) {
    while (hdr->seq.load(std::memory_order_acquire) != expected_gen) {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
}

// ---------------------------------------------------------------------------
// Blocking send/recv — used directly when ordering is already guaranteed
// ---------------------------------------------------------------------------

void shm_send(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(dest, nbytes);
    ShmDirState& st = ch.send_state;

    // Backpressure before touching the slot.
    wait_for_slot(st);

    sync_cuda_if_needed(tensor);

    // Copy the payload + stamp the header.
    uint32_t slot = st.write_slot;
    uint64_t gen  = st.next_gen;

    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.send_slot_data(slot),
                                                {tensor.numel()}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);

    ShmSlotHeader* hdr = ch.send_slot_header(slot);
    hdr->nbytes = nbytes;
    // Release-store: payload and nbytes become visible BEFORE the seq update
    // on the peer's core.  The receiver's acquire-load of seq is the pair.
    hdr->seq.store(gen, std::memory_order_release);

    // Advance sender state before posting the signal, so any subsequent
    // call sees the next slot/gen without waiting on MPI.
    st.write_slot = (slot + 1) % SHM_NUM_SLOTS;
    st.next_gen   = gen + 1;

    ShmSignal sig{gen, slot, 0};
    int err = MPI_Send(&sig, sizeof(sig), MPI_BYTE, dest, SHM_CTRL_TAG,
                       MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_send: MPI_Send signal failed: " +
                                 std::to_string(err));
}

void shm_recv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    ShmSignal sig{};
    MPI_Status status;
    int err = MPI_Recv(&sig, sizeof(sig), MPI_BYTE, source, SHM_CTRL_TAG,
                       MPI_COMM_WORLD, &status);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_recv: MPI_Recv signal failed: " +
                                 std::to_string(err));

    ShmSlotHeader* hdr = ch.recv_slot_header(sig.slot_idx);
    wait_for_gen(hdr, sig.gen);

    if (hdr->nbytes != nbytes)
        throw std::runtime_error("shm_recv: expected " + std::to_string(nbytes) +
                                 " bytes but sender wrote " +
                                 std::to_string(hdr->nbytes));

    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.recv_slot_data(sig.slot_idx),
                                                {tensor.numel()}, cpu_opts);
    tensor.copy_(shm_tensor);
    sync_cuda_if_needed(tensor);

    // Release the slot so the sender's next reuse of it can proceed.
    // fetch_add with release so our memcpy out becomes visible before the
    // counter bump from the sender's next acquire-load.
    ch.reader_done(ch.recv_half())->fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Truly async send — memcpy + slot bookkeeping happen eagerly, then a
// non-blocking MPI_Isend posts the ShmSignal.  The sender has nothing left
// to do at mpi_wait time.
// ---------------------------------------------------------------------------

MpiRequest shm_isend(torch::Tensor& tensor, int dest) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(dest, nbytes);
    ShmDirState& st = ch.send_state;

    wait_for_slot(st);

    sync_cuda_if_needed(tensor);

    uint32_t slot = st.write_slot;
    uint64_t gen  = st.next_gen;

    auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU);
    torch::Tensor shm_tensor = torch::from_blob(ch.send_slot_data(slot),
                                                {tensor.numel()}, cpu_opts);
    shm_tensor.copy_(tensor);
    if (tensor.is_cuda()) sync_cuda_if_needed(tensor);

    ShmSlotHeader* hdr = ch.send_slot_header(slot);
    hdr->nbytes = nbytes;
    hdr->seq.store(gen, std::memory_order_release);

    st.write_slot = (slot + 1) % SHM_NUM_SLOTS;
    st.next_gen   = gen + 1;

    // Post the signal non-blocking.  shm_sig inside the returned MpiRequest
    // provides stable storage for the ShmSignal buffer — MPI may read it
    // right up until MPI_Wait completes, so we can't stack-allocate it.
    MpiRequest req;
    req.shm_sig = ShmSignal{gen, slot, 0};
    int err = MPI_Isend(&req.shm_sig, sizeof(req.shm_sig), MPI_BYTE,
                        dest, SHM_CTRL_TAG, MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_isend: MPI_Isend signal failed: " +
                                 std::to_string(err));
    return req;  // on_complete is empty: nothing to do on the send side
}

// ---------------------------------------------------------------------------
// Truly async recv — post the signal irecv; the copy-out is deferred to the
// on_complete callback which mpi_wait / mpi_waitall fire after MPI_Wait
// confirms the signal landed.
// ---------------------------------------------------------------------------

MpiRequest shm_irecv(torch::Tensor& tensor, int source) {
    if (!tensor.is_contiguous()) tensor = tensor.contiguous();

    auto [mpi_dtype, typesize] = get_typesize(tensor.scalar_type());
    size_t nbytes = static_cast<size_t>(tensor.numel()) * typesize;

    ShmChannel& ch = shm_get_channel(source, nbytes);

    MpiRequest req;

    // Capture everything the callback needs.  tensor is a refcounted handle
    // (safe to copy); &ch is stable until shm_cleanup() which runs AFTER
    // all waits; nbytes is a plain value; &req.shm_sig must be read after
    // MPI_Wait fills it, so we capture by reference.
    ShmSignal* sig_ptr = &req.shm_sig;
    req.on_complete = [tensor, &ch, nbytes, sig_ptr]() mutable {
        ShmSlotHeader* hdr = ch.recv_slot_header(sig_ptr->slot_idx);
        wait_for_gen(hdr, sig_ptr->gen);

        if (hdr->nbytes != nbytes)
            throw std::runtime_error("shm_irecv: expected " +
                                     std::to_string(nbytes) +
                                     " bytes but sender wrote " +
                                     std::to_string(hdr->nbytes));

        auto cpu_opts = torch::TensorOptions().dtype(tensor.dtype())
                                              .device(torch::kCPU);
        torch::Tensor shm_tensor = torch::from_blob(
            ch.recv_slot_data(sig_ptr->slot_idx),
            {tensor.numel()}, cpu_opts);
        tensor.copy_(shm_tensor);
        sync_cuda_if_needed(tensor);

        ch.reader_done(ch.recv_half())->fetch_add(1, std::memory_order_release);
    };

    int err = MPI_Irecv(&req.shm_sig, sizeof(req.shm_sig), MPI_BYTE,
                        source, SHM_CTRL_TAG, MPI_COMM_WORLD, &req.mpi_req);
    if (err != MPI_SUCCESS)
        throw std::runtime_error("shm_irecv: MPI_Irecv signal failed: " +
                                 std::to_string(err));
    return req;
}
