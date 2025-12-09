from flask import Blueprint, jsonify
from personas.loader import load_personas

bp = Blueprint("personas", __name__, url_prefix="/api")


@bp.route("/personas", methods=["GET"])
def get_personas():
    personas = load_personas()
    return jsonify([p.to_dict() for p in personas])
