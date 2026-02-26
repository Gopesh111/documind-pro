import os
import pytest
from api.middleware import ZeroRetentionMiddleware
from fastapi import Request

@pytest.mark.asyncio
async def test_zero_retention_policy():
    """
    Test to ensure that temporary files are purged after request lifecycle.
    """
    temp_dir = "./temp_uploads"
    test_file = os.path.join(temp_dir, "sensitive_doc.txt")
    
    # Setup: Create a fake sensitive file
    os.makedirs(temp_dir, exist_ok=True)
    with open(test_file, "w") as f:
        f.write("This is sensitive user data.")
    
    assert os.path.exists(test_file)

    # Simulate Middleware Call
    middleware = ZeroRetentionMiddleware()
    
    # Mocking the 'call_next' to simulate a completed request
    async def mock_call_next(request):
        return "Response"

    await middleware(request=None, call_next=mock_call_next)

    # Verification: The file/folder should be cleared
    # In our middleware, we rmtree and makedirs, so the file should be gone
    assert not os.path.exists(test_file), "Security Breach: Sensitive file was not purged."