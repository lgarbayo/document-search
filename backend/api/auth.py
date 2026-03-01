import jwt
import datetime
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "super_secreto_hackathon" # En producción iría en tu .env
security = HTTPBearer()

# Usuarios de prueba que representan los tres niveles del RBAC
# Roles: admin | editor | lector
USERS = {
    "admin":   {"password": pwd_context.hash("admin123"),   "role": "admin"},
    "editor":  {"password": pwd_context.hash("editor123"),  "role": "editor"},
    "lector":  {"password": pwd_context.hash("lector123"),  "role": "lector"},
    # Alias legado para no romper integraciones existentes
    "empleado": {"password": pwd_context.hash("normal123"), "role": "lector"},
}

def create_token(username: str, role: str):
    # El payload es el cuerpo del token que contiene los datos del usuario
    payload = {
        "sub": username, 
        "role": role, 
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=4)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload # Retorna el diccionario con 'sub' y 'role'
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


def require_admin(current_user: dict = Security(get_current_user)):
    """Dependency que lanza 403 si el rol no es admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Solo los administradores pueden realizar esta acción")
    return current_user


def require_admin_or_editor(current_user: dict = Security(get_current_user)):
    """Dependency que lanza 403 si el rol no es admin ni editor."""
    if current_user.get("role") not in ("admin", "editor"):
        raise HTTPException(status_code=403, detail="Se requiere rol editor o superior")
    return current_user