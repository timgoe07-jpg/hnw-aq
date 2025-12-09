from flask import Blueprint, jsonify
from case_studies.loader import load_case_studies

bp = Blueprint("case_studies", __name__, url_prefix="/api")


@bp.route("/case-studies", methods=["GET"])
def get_case_studies():
    case_studies = load_case_studies()
    return jsonify([cs.to_dict() for cs in case_studies])
