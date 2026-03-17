import aiohttp

session = None


def get_global_session():
    global session
    if session is None:
        # Configure timeout for long-running LLM requests
        # - total: maximum time for the entire request (5 minutes)
        # - connect: time to establish connection (30 seconds)
        # - sock_read: time to read each chunk (5 minutes for streaming)
        timeout = aiohttp.ClientTimeout(
            total=None,  # No total timeout (let sock_read handle it)
            connect=30,  # 30 seconds to connect
            sock_read=300,  # 5 minutes per read operation (for streaming chunks)
        )
        session = aiohttp.ClientSession(timeout=timeout)
    return session