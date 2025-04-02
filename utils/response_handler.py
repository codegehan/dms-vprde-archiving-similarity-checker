from flask import jsonify
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger('response_handler')

class ResponseHandler:
    """
    A utility class to standardize API responses across the application.
    Provides consistent formatting for success and error responses.
    """
    
    @staticmethod
    def success(data=None, message="Operation successful", status_code=200):
        """
        Generate a standardized success response
        
        Args:
            data: The data to return (optional)
            message: Success message (optional)
            status_code: HTTP status code (default: 200)
            
        Returns:
            A tuple containing the JSON response and status code
        """
        response = {
            "success": True,
            "message": message,
            "status_code": status_code
        }
        
        if data is not None:
            response["data"] = data
            
        return jsonify(response), status_code
    
    @staticmethod
    def error(message="An error occurred", status_code=500, error_code=None):
        """
        Generate a standardized error response
        
        Args:
            message: Error message
            status_code: HTTP status code (default: 500)
            error_code: Application-specific error code (optional)
            details: Additional error details (optional)
            
        Returns:
            A tuple containing the JSON response and status code
        """
        response = {
            "success": False,
            "message": message,
            "status_code": status_code
        }
        
        if error_code:
            response["error_code"] = error_code
            
        return jsonify(response), status_code
    
    @staticmethod
    def handle_exception(e, custom_message=None):
        """
        Handle exceptions and generate appropriate error responses
        
        Args:
            e: The exception that was raised
            custom_message: Optional custom message to override default
            
        Returns:
            A tuple containing the JSON response and status code
        """
        # Log the exception
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Determine status code based on exception type
        if isinstance(e, ValueError):
            status_code = 400  # Bad Request
        elif isinstance(e, PermissionError):
            status_code = 403  # Forbidden
        elif isinstance(e, FileNotFoundError):
            status_code = 404  # Not Found
        else:
            status_code = 500  # Internal Server Error
        
        # Use custom message if provided, otherwise use exception message
        message = custom_message if custom_message else str(e)
        
        # Get exception details for development environments
        # In production, you might want to disable this
        # details = {
        #     "exception_type": e.__class__.__name__,
        #     "traceback": traceback.format_exc().split("\n")
        # }
        
        return ResponseHandler.error(
            message=message,
            status_code=status_code,
            # details=details
        )