# Direct Mode vLLM Tests

## ⚠️ IMPORTANT: Docker Required

**ALL Direct mode tests MUST be run inside the Docker container.**

Direct mode tests cannot run locally because they require:
- Full vLLM build with C++ extensions
- Compiled CUDA/CPU kernels
- Proper library paths and dependencies
- The `vllm._C` module which is only available in the built container

## Running Direct Mode Tests

### Run All Direct Mode Tests
```bash
docker run --rm -v $(pwd):/app scalarlm-cray:latest \
    python -m pytest /app/test/integration/direct/ -v
```

### Run Specific Test
```bash
docker run --rm -v $(pwd):/app scalarlm-cray:latest \
    python -m pytest /app/test/integration/direct/test_direct_mode_coverage.py -v
```

### Debug Mode
```bash
docker run --rm -it -v $(pwd):/app scalarlm-cray:latest \
    python -m pytest /app/test/integration/direct/ -vvs --pdb
```

## Test Files

- `test_direct_mode_coverage.py` - Comprehensive Direct mode functionality tests
- `test_direct_mode_fixed.py` - Fixed Direct mode tests with proper error handling

## Common Errors

### "No module named 'vllm._C'"
**Cause**: Trying to run Direct mode tests locally instead of in Docker
**Solution**: Use the Docker commands above

### "vLLM not built with CPU support"
**Cause**: Using wrong Docker image
**Solution**: Ensure you're using `scalarlm-cray:latest` which has CPU support

### Import errors for vLLM modules
**Cause**: Running outside Docker or using outdated image
**Solution**: Rebuild Docker image: `docker-compose build cray`

## Writing New Direct Mode Tests

When adding new Direct mode tests:

1. Place them in this directory (`test/integration/direct/`)
2. Always include a docstring explaining what the test covers
3. Add Docker requirement comments at the top of the file
4. Use proper async patterns for AsyncLLM tests

Example:
```python
"""
Test Direct mode vLLM engine functionality.
REQUIRES: Docker container with full vLLM build
"""
import pytest
from vllm import LLM  # Only works in Docker

def test_direct_mode_initialization():
    """Test that Direct mode LLM initializes correctly."""
    llm = LLM(model="facebook/opt-125m", mode="direct")
    assert llm is not None
```