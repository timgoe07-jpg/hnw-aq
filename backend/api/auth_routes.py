import secrets
import urllib.parse
import requests
from flask import Blueprint, jsonify, request, session
from config import Config

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _build_auth_url(state: str) -> str:
    if not Config.LINKEDIN_CLIENT_ID or not Config.LINKEDIN_REDIRECT_URI:
        raise ValueError("LinkedIn client config missing (set LINKEDIN_CLIENT_ID and LINKEDIN_REDIRECT_URI)")
    params = {
        "response_type": "code",
        "client_id": Config.LINKEDIN_CLIENT_ID,
        "redirect_uri": Config.LINKEDIN_REDIRECT_URI,
        # OpenID Connect flow for consumer Sign-In (per LinkedIn docs)
        "scope": "openid profile email",
        "state": state,
    }
    return "https://www.linkedin.com/oauth/v2/authorization?" + urllib.parse.urlencode(params)


def _exchange_code_for_token(code: str) -> dict:
    if not Config.LINKEDIN_CLIENT_ID or not Config.LINKEDIN_CLIENT_SECRET:
        raise ValueError("LinkedIn client ID/secret not configured")
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": Config.LINKEDIN_REDIRECT_URI,
        "client_id": Config.LINKEDIN_CLIENT_ID,
        "client_secret": Config.LINKEDIN_CLIENT_SECRET,
    }
    resp = requests.post("https://www.linkedin.com/oauth/v2/accessToken", data=data, timeout=10)
    resp.raise_for_status()
    return resp.json()


@bp.route("/linkedin/login", methods=["GET"])
def linkedin_login():
    try:
        state = secrets.token_urlsafe(16)
        session["oauth_state"] = state
        url = _build_auth_url(state)
        return jsonify({"url": url})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@bp.route("/linkedin/callback", methods=["GET"])
def linkedin_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    error = request.args.get("error_description") or request.args.get("error")
    if error:
        return jsonify({"error": error}), 400
    if not code or not state or state != session.get("oauth_state"):
        return jsonify({"error": "Invalid state or code"}), 400
    try:
        token_data = _exchange_code_for_token(code)
        session["linkedin_access_token"] = token_data.get("access_token")
        session["linkedin_expires_in"] = token_data.get("expires_in")
        return jsonify({"status": "connected"})
    except Exception as exc:
        return jsonify({"error": f"Token exchange failed: {exc}"}), 400
