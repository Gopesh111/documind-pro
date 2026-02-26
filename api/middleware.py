import os
import shutil
import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZeroRetentionMiddleware:
    """
    Ensures that any temporary files or buffers are cleared 
    immediately after the request is processed.
    """
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Logic to trigger cleanup of temp upload directories
        temp_dir = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
                logger.info("Security: Temporary session data purged (Zero-Retention Policy).")
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                
        return response