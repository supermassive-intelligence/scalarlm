from masint.util.make_api_url import make_api_url

from masint.engines.cray.submit_training_job import submit_training_job
from masint.engines.cray.submit_slurm_job import submit_slurm_job

import aiohttp
import aiofiles
import contextlib
import json
import tempfile

import logging

logger = logging.getLogger(__name__)


class AsyncCray:
    def __init__(self, api_url=None):
        self.api_url = api_url

    async def train(self, data, model_name, train_args):
        return await submit_training_job(data, model_name, train_args, api_url=self.api_url)

    async def submit_slurm_job(self, code, train_args=None):
        return await submit_slurm_job(code, train_args, api_url=self.api_url)

    async def generate(self, prompts, model_name, max_tokens):

        upload_threshold = 128

        if len(prompts) > upload_threshold:
            result = await upload_generate(prompts, model_name, max_tokens, api_url=self.api_url)

            handle_error(result)

            final_result = await poll_for_downloads(result, api_url=self.api_url)
        else:
            result = await self.submit_generate(prompts, model_name, max_tokens)

            handle_error(result)

            final_result = await poll_for_responses(result, api_url=self.api_url)

        return [response["response"] for response in final_result["results"]]

    async def submit_generate(self, prompts, model_name, max_tokens):
        api_url = make_api_url("v1/generate", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            params = {"prompts": prompts}

            if model_name is not None:
                params["model"] = model_name

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            async with session.post(api_url, json=params) as resp:
                assert resp.status == 200
                return await resp.json()

    async def get_results(self, request_ids):
        async with aiohttp.ClientSession() as session:
            api_url = make_api_url("v1/generate/get_results", api_url=self.api_url)
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                return await resp.json()

    async def list_models(self):
        api_url = make_api_url("v1/megatron/list_models", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def get_training_job(self, job_dir):
        api_url = make_api_url(f"v1/megatron/train/{job_dir}", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def health(self):
        api_url = make_api_url("v1/health", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def metrics(self):
        api_url = make_api_url("v1/generate/metrics", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                return await resp.json()

    async def get_gpu_count(self):
        api_url = make_api_url("v1/megatron/gpu_count", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                response = await resp.json()
                logger.debug(f"get_gpu_count response: {response}")
                return response["gpu_count"]

    async def get_node_count(self):
        api_url = make_api_url("v1/megatron/node_count", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as resp:
                response = await resp.json()
                logger.debug(f"get_node_count response: {response}")
                return response["node_count"]

    async def cancel(self, model_name):
        api_url = make_api_url(f"v1/megatron/cancel/{model_name}", api_url=self.api_url)
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url) as resp:
                return await resp.json()

    async def clear_queue(self):
        api_url = make_api_url("v1/generate/clear_queue", api_url=self.api_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(api_url) as resp:
                return await resp.json()

def handle_error(result):
    if "error" in result and result["error"] is not None:
        logger.error(f"Error in response: {result['error']}")
        raise Exception(result["error"])

    if not result.get("results"):
        logger.error(f"No results found in response: {result}")
        raise Exception("No results found in response")

    if not isinstance(result["results"], list):
        logger.error(f"Results is not a list: {result['results']}")
        raise Exception("Results is not a list")


async def poll_for_responses(result, api_url):
    api_url = make_api_url("v1/generate/get_results", api_url=api_url)

    async with aiohttp.ClientSession() as session:
        while not is_finished(result):
            request_ids = [response["request_id"] for response in result["results"]]
            async with session.post(api_url, json={"request_ids": request_ids}) as resp:
                assert resp.status == 200
                result = await resp.json()

            handle_error(result)

    return result


def is_finished(result):
    for response in result["results"]:
        if response["error"] is not None:
            raise Exception(response["error"])

        if response["response"] is None:
            return False

    return True

async def upload_generate(prompts, model_name, max_tokens, api_url=None):

    with make_upload_json_file(prompts, model_name, max_tokens) as upload_path:

        api_url = make_api_url("v1/generate/upload", api_url=api_url)

        return await upload_async(upload_path, api_url)

@contextlib.contextmanager
def make_upload_json_file(prompts, model_name, max_tokens):
    requests_object = {
        "prompts": prompts,
        "model_name": model_name,
        "max_tokens": max_tokens
    }

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
        json.dump(requests_object, f)
        f.flush()
        f.seek(0)

        logger.debug(f"Created temporary upload file at {f.name}")
        logger.debug(f"Upload file size: {f.tell()} bytes")

        yield f.name


async def upload_async(data_file_path, api_url):
    async with aiohttp.ClientSession() as session:

        content_length = await get_content_length(data_file_path)

        with make_multipart_writer(data_file_path) as mp:

            headers = mp.headers

            headers["Content-Length"] = str(content_length)

            async with session.post(api_url, data=mp, headers=headers) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to upload data: {await resp.text()}")
                return await resp.json()

async def get_content_length(data_file_path):
    with make_multipart_writer(data_file_path) as mp:

        class Writer:
            def __init__(self):
                self.count = 0

            async def write(self, data):
                self.count += len(data)

        writer = Writer()
        await mp.write(writer)
        content_length = writer.count

        return content_length

@contextlib.contextmanager
def make_multipart_writer(data_file_path):
    with aiohttp.MultipartWriter("form-data") as mp:
        part = mp.append(file_sender(data_file_path))
        part.set_content_disposition("form-data", name="file", filename="requests.json")

    yield mp

async def file_sender(file_path):
    chunk_size = 64 * 1024

    async with aiofiles.open(file_path, "rb") as f:
        chunk = await f.read(chunk_size)
        index = 0
        while chunk:
            yield chunk
            chunk = await f.read(chunk_size)
            index += chunk_size

async def poll_for_downloads(result, api_url=None):
    api_url = make_api_url("v1/generate/download", api_url=api_url)

    request_id = result["request_id"]

    async with aiohttp.ClientSession() as session:
        final_result = None

        while not is_download_finished(final_result):
            async with session.post(api_url, json={"request_id": request_id}) as response:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with tempfile.NamedTemporaryFile(mode="w+", delete=True) as f:
                    logger.debug(f"Created temporary download file at {f.name}")

                    async for chunk in response.content.iter_chunked(64 * 1024):
                        if chunk:
                            f.write(chunk.decode('utf-8'))
                            downloaded += len(chunk)
                            logger.debug(f"Downloaded {downloaded} of {total_size} bytes ({(downloaded/total_size)*100:.2f}%)")

                    f.flush()
                    f.seek(0)

                    final_result = json.load(f)

    return final_result

def is_download_finished(result):
    if result is None:
        return False

    logger.debug(f"Download result status: {result['status']}")

    if result["status"] != "completed":
        return False

    for response in result["results"]:
        if response["error"] is not None:
            raise Exception(response["error"])

    return True
