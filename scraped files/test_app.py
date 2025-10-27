"""from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>Home Page</h1>"

@app.get("/about", response_class=HTMLResponse)
async def about():
    return "<h1>About Page</h1>"
"""