import logging

import uvicorn

HOST = "127.0.0.1"
PORT = 8000

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("citewise.api:app", host=HOST, port=PORT)
