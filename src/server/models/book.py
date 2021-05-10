from flask_restplus import fields
from src.server.instance import server

products = server.api.model("Products", {
    "id": fields.String(description="ID registro"),
    "title": fields.String(required=True, 
    min_Length=1, max_Length=200, description="t")
})
