def error(message: str, data: dict = None) -> dict:
    """
    Creates a response for an error situation.

    Args:
    - message (str): A description of the error.
    - data (dict, optional): Additional data to include in the response.

    Returns:
    - dict: A dictionary containing the error response.
    """
    response = {
        'status': 'error',
        'message': message,
        'data': data or {}
    }
    return response

def success(message: str, data: dict = None) -> dict:
    """
    Creates a response for a successful operation.

    Args:
    - message (str): A description of the success.
    - data (dict, optional): Additional data to include in the response.

    Returns:
    - dict: A dictionary containing the success response.
    """
    response = {
        'status': 'success',
        'message': message,
        'data': data or {}
    }
    return response
