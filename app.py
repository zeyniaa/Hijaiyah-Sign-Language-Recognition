"""Start the project's local browser workspace with python app.py."""
import os
from web_app import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('PORT', '7861')))
