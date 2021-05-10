from flask_restplus import fields
from src.server.instance import server

products = server.api.model("Products", {
    "imagem_galpao": fields.String(description="Imagem galpao",required=True),
    "imagens_products": fields.String(description="Imagens produtos", required=True),
    "marcas" : fields.String(description="Nomes dos produtos", required=True),
})