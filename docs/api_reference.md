# API Reference

This section provides detailed information about the API endpoints available in this project.

## Endpoints

### GET /api/v1/example

**Description:** Retrieves a list of examples.

**Request:**

```http
GET /api/v1/example HTTP/1.1
Host: api.example.com
```

**Response:**
```json
[
    {
        "id": 1,
        "name": "Example 1",
        "description": "This is an example."
    },
    {
        "id": 2,
        "name": "Example 2",
        "description": "This is another example."
    }
]
```

---

### POST /api/v1/example

**Description:** Creates a new example.

**Request:**

```http
POST /api/v1/example HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
    "name": "New Example",
    "description": "This is a new example."
}
```

**Response:**
```json
{
    "id": 3,
    "name": "New Example",
    "description": "This is a new example."
}
```

---

### PUT /api/v1/example/

**Description:** Updates a specific example by ID.

**Request:**

```http
PUT /api/v1/example/1 HTTP/1.1
Host: api.example.com
Content-Type: application/json

{
    "name": "Updated Example",
    "description": "This is an updated example."
}
```

**Response:**
```json
{
    "id": 1,
    "name": "Updated Example",
    "description": "This is an updated example."
}
```

---

### DELETE /api/v1/example/

**Description:** Deletes a specific example by ID.

**Request:**

```http
DELETE /api/v1/example/1 HTTP/1.1
Host: api.example.com
```

**Response:**
```json
{
    "message": "Example deleted successfully."
}
```
