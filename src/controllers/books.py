from flask import Flask
from flask_restplus import Api, Resource

from src.server.instance import server
from src.models.products import products
from src.models_detect.image_detect import detect_products

app, api = server.app, server.api

books_db = [
    {"id":0, "title":"jk"}

]

@api.route("/products")
class ProductsList(Resource):
    #@api.marshal_list_with(book)
    #def get(self,):
    #    return books_db


    #@api.expect(book, validate=True)
    #@api.marshal_with(book)
    def post(self,):
        response = api.payload
        if "imagem_galpao" not in response:
            return generate_response(400,"O nome image_galpao é obrigatorio"), 400

        if "imagens_products" not in response:
            return generate_response(400,"O nome image_galpao é obrigatorio"), 400
        
        if "marcas" not in response:
            return generate_response(400,"O nome marcas é obrigatorio"), 400

        result_products = detect_products(response["imagem_galpao"], response["imagens_products"], response["marcas"])

        return generate_response(200, "Imagem recebida", "results", result_products), 200

def generate_response(status, message, name_value=False, value=False):
    response = {}
    response["status"] = status
    response["message"] = message
    if name_value and value:
        response[name_value] = value
    return response