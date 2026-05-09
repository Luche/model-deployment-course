"""
FastAPI Fundamentals — Session 06
Covers: path params, query params, request body, form data, file uploads,
        headers, cookies, request object, response customization, errors.

Run:  uv run uvicorn first_api:app --reload
Docs: http://localhost:8000/docs
"""

from typing import Optional

from fastapi import (
    Body, Cookie, FastAPI, File, Form, Header,
    HTTPException, Path, Query, Request, Response, UploadFile, status,
)
from fastapi.responses import (
    HTMLResponse, JSONResponse,
    PlainTextResponse,
)

app = FastAPI(title="FastAPI Fundamentals")


# ==============================================================================
# 1. PATH PARAMETERS
# ==============================================================================

# 1a. Basic path parameter with type coercion
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """FastAPI converts {user_id} from string to int automatically."""
    return {"user_id": user_id}


# 1b. Multiple path parameters
@app.get("/users/{user_type}/{user_id}")
async def get_typed_user(user_type: str, user_id: int):
    return {"user_type": user_type, "user_id": user_id}


# 1c. Path validation with Path()
@app.get("/items/{item_id}")
async def get_item(item_id: int = Path(..., ge=1, le=1000)):
    """
    Path(...) lets you add validation:
      ge=1    → must be >= 1
      le=1000 → must be <= 1000
    '...' means the parameter is required (no default).
    """
    return {"item_id": item_id}


# 1d. String path validation
@app.get("/license-plates/{license}")
async def get_license_plate(
    license: str = Path(..., min_length=5, max_length=11, pattern=r"^[A-Z]{1,2}-\d{1,4}-[A-Z]{1,3}$")
):
    """Validates length AND format with a regex pattern."""
    return {"license": license}


# ==============================================================================
# 2. QUERY PARAMETERS
# ==============================================================================

# 2a. Basic query params — everything not in the path becomes a query param
@app.get("/search")
async def search(
    q: str,                         # required
    page: int = 1,                  # optional with default
    size: int = 10,                 # optional with default
    active: Optional[bool] = None,  # fully optional (omittable)
):
    """
    URL example: /search?q=python&page=2&size=5&active=true
    FastAPI converts types automatically (e.g. "true" → True).
    """
    return {"q": q, "page": page, "size": size, "active": active}


# 2b. Query validation with Query()
@app.get("/paginate")
async def paginate(
    page: int = Query(1, gt=0),     # must be > 0
    size: int = Query(10, le=100),  # must be <= 100
):
    """Query() adds the same validation constraints as Path()."""
    return {"page": page, "size": size}


# 2c. Path + query together
@app.get("/users/{user_id}/orders")
async def get_user_orders(user_id: int, status_filter: Optional[str] = None):
    return {"user_id": user_id, "status_filter": status_filter}


# ==============================================================================
# 3. REQUEST BODY
# ==============================================================================

# 3a. Body() scalars — read individual JSON fields from the request body
@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(
    name: str = Body(...),
    price: float = Body(...),
    in_stock: bool = Body(True),
):
    """
    Client sends JSON:  {"name": "Laptop", "price": 999.99}
    Body(...) = required field.  Body(True) = optional with default.
    """
    return {"name": name, "price": price, "in_stock": in_stock}


# 3b. Multiple independent Body() fields
@app.post("/login")
async def login(username: str = Body(...), password: str = Body(...)):
    """'...' means required — FastAPI will reject the request if missing."""
    return {"username": username}


# 3c. Body() fields with validation constraints
@app.post("/items/prioritized")
async def create_prioritized_item(
    name: str = Body(...),
    price: float = Body(..., gt=0),     # must be > 0
    priority: int = Body(..., ge=1, le=3),  # must be 1, 2, or 3
):
    return {"name": name, "price": price, "priority": priority}


# 3d. Path param + body (update scenario)
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    name: str = Body(...),
    price: float = Body(...),
):
    return {"item_id": item_id, "name": name, "price": price}


# ==============================================================================
# 4. FORM DATA & FILE UPLOADS
# ==============================================================================

# 4a. Form fields (HTML <form> / application/x-www-form-urlencoded)
@app.post("/register")
async def register(name: str = Form(...), age: int = Form(...)):
    """
    Form() reads from form-encoded data, not JSON.
    Use this for HTML forms or multipart requests.
    """
    return {"name": name, "age": age}


# 4b. File upload — entire file read into memory as bytes
@app.post("/upload/bytes")
async def upload_bytes(file: bytes = File(...)):
    """Reads the whole file into memory. Fine for small files."""
    return {"file_size": len(file)}


# 4c. File upload — streaming with UploadFile (preferred for large files)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    UploadFile gives you:
      file.filename     — original filename
      file.content_type — MIME type
      await file.read() — read content
    """
    return {"filename": file.filename, "content_type": file.content_type}


# 4d. Multiple file uploads
@app.post("/upload/multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    return [{"filename": f.filename, "content_type": f.content_type} for f in files]


# ==============================================================================
# 5. HEADERS & COOKIES
# ==============================================================================

# 5a. Reading request headers
@app.get("/headers")
async def read_headers(
    user_agent: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None),
):
    """
    Header() reads request headers.
    FastAPI auto-converts hyphens → underscores: X-Token → x_token.
    Headers are case-insensitive in HTTP.
    """
    return {"user_agent": user_agent, "x_token": x_token}


# 5b. Reading cookies
@app.get("/profile")
async def read_profile(session_id: Optional[str] = Cookie(None)):
    """Cookie() reads a named cookie from the request."""
    return {"session_id": session_id}


# 5c. Setting a cookie in the response
@app.post("/session/start")
async def start_session(response: Response):
    """
    Inject Response as a parameter to mutate the outgoing response
    (set cookies, headers) while still returning a normal dict.
    """
    response.set_cookie(key="session_id", value="abc1234", httponly=True, max_age=86400)
    return {"message": "Session started"}


# ==============================================================================
# 6. THE RAW REQUEST OBJECT
# ==============================================================================

@app.get("/inspect")
async def inspect_request(request: Request):
    """
    The Request object gives you everything about the incoming HTTP request:
    URL, headers, query string, client IP, etc.
    Useful when you need something FastAPI doesn't expose directly.
    """
    return {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "client_ip": request.client.host,
        "headers": dict(request.headers),
    }


# ==============================================================================
# 7. RESPONSE CUSTOMIZATION
# ==============================================================================

# Dummy in-memory "database" (plain dict)
_posts_db = {
    1: {"title": "Hello FastAPI", "body": "Getting started..."},
}

# 7o. Basic response with custom status code
@app.get("/posts/{post_id}")
async def read_post(post_id: int):
    post = _posts_db.get(post_id)
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    return {"id": post_id, **post}

# 7a. Static status code on the decorator
@app.post("/posts", status_code=status.HTTP_201_CREATED)
async def create_post(title: str = Body(...), body: str = Body(...)):
    """status_code on the decorator sets the success status to 201 Created."""
    new_id = max(_posts_db) + 1
    _posts_db[new_id] = {"title": title, "body": body}
    return {"id": new_id, "title": title, "body": body}


# 7b. 204 No Content — body must be empty
@app.delete("/posts/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(post_id: int):
    _posts_db.pop(post_id, None)
    # Return None — FastAPI sends an empty body with 204


# 7c. Dynamic status code via Response parameter
@app.put("/posts/{post_id}")
async def upsert_post(
    post_id: int,
    title: str = Body(...),
    body: str = Body(...),
    response: Response = None,
):
    """
    Sets 201 if the resource was created, 200 (default) if updated.
    Inject Response to change the status code at runtime.
    """
    if post_id not in _posts_db:
        response.status_code = status.HTTP_201_CREATED
    _posts_db[post_id] = {"title": title, "body": body}
    return {"id": post_id, "title": title, "body": body}


# 7d. Partial update with PATCH
@app.patch("/posts/{post_id}")
async def patch_post(
    post_id: int,
    title: Optional[str] = Body(None),
    body: Optional[str] = Body(None),
):
    """
    PATCH only updates the fields you send — missing fields keep their current value.
    Compare with PUT above which replaces the whole resource.
    """
    if post_id not in _posts_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Post not found")
    if title is not None:
        _posts_db[post_id]["title"] = title
    if body is not None:
        _posts_db[post_id]["body"] = body
    return {"id": post_id, **_posts_db[post_id]}


# 7e. Adding custom headers to the response
@app.get("/custom-headers")
async def custom_headers():
    content = {"message": "See the response headers in your HTTP client"}
    headers = {"X-Request-Id": "abc-123", "X-Powered-By": "FastAPI"}
    return JSONResponse(content=content, headers=headers)


# 7e. Plain text response
@app.get("/text", response_class=PlainTextResponse)
async def plain_text():
    return "This is plain text, not JSON."


# 7f. HTML response
@app.get("/html", response_class=HTMLResponse)
async def html_page():
    return """
    <html>
      <head><title>FastAPI</title></head>
      <body><h1>Hello from FastAPI!</h1></body>
    </html>
    """


# ==============================================================================
# 8. ERROR HANDLING
# ==============================================================================

# 8a. Simple HTTPException
@app.get("/secure")
async def secure_endpoint(x_token: str = Header(...)):
    if x_token != "secret":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return {"message": "Welcome"}


# 8b. HTTPException with structured detail
@app.post("/password-check")
async def check_password(
    password: str = Body(...),
    password_confirm: str = Body(...),
):
    """
    raise HTTPException stops execution and returns the error immediately.
    'detail' can be a string or any JSON-serializable object.
    """
    if password != password_confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Passwords don't match",
                "hints": ["Check caps lock", "Use the eye icon to reveal your typing"],
            },
        )
    return {"message": "Passwords match"}
