"""Simulated tool results and file contents for context growth."""

# ---------------------------------------------------------------------------
# Simulated tool results (KEY for realistic context growth)
# ---------------------------------------------------------------------------

SIMULATED_SEARCH_RESULT = """\
search_files("session token validation")
Found 12 matches across 5 files:

app/models.py:45:    session_token = models.CharField(max_length=255, unique=True, db_index=True)
app/models.py:78:    def validate_session_token(cls, token: str) -> bool:
app/models.py:82:        if not token or len(token) < 16:
app/models.py:90:            return cls._check_token_format(token)
app/views.py:23:from app.models import User, Session, validate_session_token
app/views.py:112:    if not validate_session_token(request.headers.get('X-Session-Token')):
app/views.py:115:        raise AuthenticationError("Invalid or expired session token")
app/views.py:189:        token = request.headers.get('X-Session-Token')
app/views.py:192:        session = Session.objects.filter(token=token).first()
app/views.py:195:        if not session or not session.is_active:
app/validators.py:34:def validate_token_format(token: str) -> Tuple[bool, Optional[str]]:
app/validators.py:56:    if not re.match(TOKEN_PATTERN, token):
app/tests/test_session.py:22:    def test_validate_session_token_valid(self):
app/tests/test_session.py:28:    def test_validate_session_token_expired(self):
app/tests/test_session.py:35:    def test_validate_session_token_malformed(self):

Relevant files:
  - app/models.py (4 matches)
  - app/views.py (5 matches)
  - app/validators.py (2 matches)
  - app/tests/test_session.py (3 matches)
  - config/settings.py (0 matches, referenced in imports)
"""

SIMULATED_FILE_CONTENTS = {
    "models.py": """\
\"\"\"
app/models.py - Data models for the authentication and session management system.

Defines User, Session, and Role models with validation logic for session tokens.
Uses Django ORM with custom managers for session lifecycle management.
\"\"\"

import hashlib
import secrets
from datetime import timedelta
from typing import Optional, ClassVar

from django.db import models
from django.utils import timezone
from django.core.exceptions import ValidationError


class Role(models.Model):
    \"\"\"User role for authorization decisions.\"\"\"
    name = models.CharField(max_length=50, unique=True)
    permissions = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "auth_roles"
        ordering = ["name"]

    def __str__(self):
        return self.name

    def has_permission(self, perm: str) -> bool:
        \"\"\"Check if this role grants the given permission.\"\"\"
        return perm in self.permissions.get("grants", [])

    def add_permission(self, perm: str) -> None:
        \"\"\"Add a permission to this role.\"\"\"
        if "grants" not in self.permissions:
            self.permissions["grants"] = []
        if perm not in self.permissions["grants"]:
            self.permissions["grants"].append(perm)
        self.save(update_fields=["permissions", "updated_at"])


class User(models.Model):
    \"\"\"Application user with authentication credentials.\"\"\"
    username = models.CharField(max_length=150, unique=True, db_index=True)
    email = models.EmailField(unique=True, db_index=True)
    password_hash = models.CharField(max_length=255)
    role = models.ForeignKey(
        Role, on_delete=models.SET_NULL, null=True, related_name="users"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Class-level constants
    MAX_ACTIVE_SESSIONS: ClassVar[int] = 5

    class Meta:
        db_table = "auth_users"
        ordering = ["username"]

    def __str__(self):
        return self.username

    def set_password(self, raw_password: str) -> None:
        \"\"\"Hash and store the password using bcrypt-like approach.\"\"\"
        salt = secrets.token_hex(16)
        self.password_hash = f"sha256${salt}${hashlib.sha256(f'{salt}{raw_password}'.encode()).hexdigest()}"
        self.save(update_fields=["password_hash", "updated_at"])

    def check_password(self, raw_password: str) -> bool:
        \"\"\"Verify the password against the stored hash.\"\"\"
        if not self.password_hash or "$" not in self.password_hash:
            return False
        algo, salt, stored_hash = self.password_hash.split("$", 2)
        computed = hashlib.sha256(f"{salt}{raw_password}".encode()).hexdigest()
        return secrets.compare_digest(computed, stored_hash)

    @property
    def active_sessions(self):
        \"\"\"Return queryset of active sessions for this user.\"\"\"
        return self.sessions.filter(is_active=True)

    def can_create_session(self) -> bool:
        \"\"\"Check if user can create a new session (within limit).\"\"\"
        return self.active_sessions.count() < self.MAX_ACTIVE_SESSIONS


class Session(models.Model):
    \"\"\"User session with token-based authentication.\"\"\"
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="sessions"
    )
    session_token = models.CharField(max_length=255, unique=True, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, default="")

    class Meta:
        db_table = "auth_sessions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "is_active"], name="idx_session_user_active"),
        ]

    def __str__(self):
        return f"Session({self.user.username}, active={self.is_active})"

    @classmethod
    def create_for_user(cls, user: User, ttl: timedelta = timedelta(hours=24)) -> "Session":
        \"\"\"Create a new session for the given user.\"\"\"
        if not user.can_create_session():
            raise ValidationError(
                f"User {user.username} has reached the maximum of "
                f"{User.MAX_ACTIVE_SESSIONS} active sessions."
            )
        token = secrets.token_urlsafe(48)
        session = cls(
            user=user,
            session_token=token,
            expires_at=timezone.now() + ttl,
        )
        session.save()
        return session

    @classmethod
    def validate_session_token(cls, token: str) -> bool:
        \"\"\"Validate a session token without race conditions.

        BUG: The original implementation checks is_active and expiry
        separately, which creates a TOCTOU race condition when multiple
        requests arrive simultaneously for the same session.
        \"\"\"
        if not token or len(token) < 16:
            return False
        try:
            session = cls.objects.select_for_update().filter(
                session_token=token
            ).first()
        except Exception:
            return False
        if not session:
            return False
        # BUG: is_active check is not atomic with the expiry check
        if not session.is_active:
            return False
        if session.expires_at < timezone.now():
            session.is_active = False
            session.save(update_fields=["is_active"])
            return False
        return cls._check_token_format(token)

    @classmethod
    def _check_token_format(cls, token: str) -> bool:
        \"\"\"Verify the token matches the expected format.\"\"\"
        import re
        return bool(re.match(r'^[A-Za-z0-9_-]{48,}$', token))

    def deactivate(self) -> None:
        \"\"\"Mark this session as inactive.\"\"\"
        self.is_active = False
        self.save(update_fields=["is_active"])

    def refresh(self, ttl: timedelta = timedelta(hours=24)) -> None:
        \"\"\"Extend the session expiry time.\"\"\"
        self.expires_at = timezone.now() + ttl
        self.save(update_fields=["expires_at"])

    def is_expired(self) -> bool:
        \"\"\"Check if the session has expired.\"\"\"
        return self.expires_at < timezone.now()
""",

    "views.py": """\
\"\"\"
app/views.py - HTTP request handlers for authentication endpoints.

Provides login, logout, session validation, and token refresh endpoints.
Uses Flask-style request handling with custom middleware for auth.
\"\"\"

import logging
from functools import wraps
from typing import Optional, Dict, Any

from django.http import JsonResponse, HttpRequest
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from app.models import User, Session, Role
from app.validators import (
    validate_token_format,
    validate_email,
    validate_password_strength,
    AuthenticationError,
    TokenValidationError,
)

logger = logging.getLogger(__name__)


def require_auth(f):
    \"\"\"Decorator that requires a valid session token in the request header.\"\"\"
    @wraps(f)
    def decorated(request: HttpRequest, *args, **kwargs):
        token = request.headers.get("X-Session-Token")
        if not token:
            return JsonResponse(
                {"error": "Missing session token", "code": "AUTH_TOKEN_MISSING"},
                status=401,
            )
        if not Session.validate_session_token(token):
            return JsonResponse(
                {"error": "Invalid or expired session token", "code": "AUTH_TOKEN_INVALID"},
                status=403,
            )
        # Attach session to request for downstream use
        session = Session.objects.filter(session_token=token).first()
        if session:
            request.session_obj = session
            request.current_user = session.user
        return f(request, *args, **kwargs)
    return decorated


@csrf_exempt
@require_http_methods(["POST"])
def login(request: HttpRequest) -> JsonResponse:
    \"\"\"Authenticate a user and create a new session.

    Request body (JSON):
        username: str
        password: str

    Returns:
        JSON with session_token on success, error on failure.
    \"\"\"
    import json
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse(
            {"error": "Invalid JSON body", "code": "INVALID_BODY"},
            status=400,
        )

    username = body.get("username", "").strip()
    password = body.get("password", "")

    if not username or not password:
        return JsonResponse(
            {"error": "Username and password are required", "code": "MISSING_CREDENTIALS"},
            status=400,
        )

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        logger.warning("Login attempt for unknown user: %s", username)
        return JsonResponse(
            {"error": "Invalid credentials", "code": "AUTH_FAILED"},
            status=401,
        )

    if not user.is_active:
        logger.warning("Login attempt for inactive user: %s", username)
        return JsonResponse(
            {"error": "Account is disabled", "code": "ACCOUNT_DISABLED"},
            status=403,
        )

    if not user.check_password(password):
        logger.warning("Failed login for user: %s", username)
        return JsonResponse(
            {"error": "Invalid credentials", "code": "AUTH_FAILED"},
            status=401,
        )

    try:
        session = Session.create_for_user(user)
    except Exception as e:
        logger.error("Failed to create session for user %s: %s", username, e)
        return JsonResponse(
            {"error": "Could not create session", "code": "SESSION_CREATE_FAILED"},
            status=500,
        )

    logger.info("User %s logged in successfully", username)
    return JsonResponse({
        "session_token": session.session_token,
        "expires_at": session.expires_at.isoformat(),
        "user": {
            "username": user.username,
            "email": user.email,
            "role": user.role.name if user.role else None,
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def logout(request: HttpRequest) -> JsonResponse:
    \"\"\"Deactivate the current session.

    Requires X-Session-Token header.
    \"\"\"
    token = request.headers.get("X-Session-Token")
    session = Session.objects.filter(session_token=token).first()
    if session:
        session.deactivate()
        logger.info("User %s logged out", session.user.username)
    return JsonResponse({"status": "logged_out"})


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def validate_session(request: HttpRequest) -> JsonResponse:
    \"\"\"Validate the current session and return user info.

    BUG: This endpoint calls validate_session_token separately from
    require_auth, creating a double-check race condition. Between the
    require_auth check and this query, the session could be deactivated
    by a concurrent logout request, leading to inconsistent state.

    Requires X-Session-Token header.
    \"\"\"
    token = request.headers.get("X-Session-Token")
    # Redundant validation -- should reuse request.session_obj instead
    if not Session.validate_session_token(token):
        return JsonResponse(
            {"error": "Session validation failed", "code": "SESSION_INVALID"},
            status=403,
        )
    session = request.session_obj
    user = request.current_user
    return JsonResponse({
        "valid": True,
        "user": {
            "username": user.username,
            "email": user.email,
            "role": user.role.name if user.role else None,
            "is_active": user.is_active,
        },
        "session": {
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "last_accessed_at": session.last_accessed_at.isoformat(),
        },
    })


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def refresh_session(request: HttpRequest) -> JsonResponse:
    \"\"\"Refresh the current session, extending its expiry time.

    Requires X-Session-Token header.
    \"\"\"
    import json
    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        body = {}

    ttl_hours = body.get("ttl_hours", 24)
    from datetime import timedelta
    ttl = timedelta(hours=max(1, min(ttl_hours, 168)))  # Cap at 7 days

    session = request.session_obj
    session.refresh(ttl)
    logger.info("Session refreshed for user %s", session.user.username)

    return JsonResponse({
        "expires_at": session.expires_at.isoformat(),
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def list_sessions(request: HttpRequest) -> JsonResponse:
    \"\"\"List all active sessions for the current user.

    Requires X-Session-Token header.
    \"\"\"
    user = request.current_user
    sessions = user.active_sessions.order_by("-created_at")
    return JsonResponse({
        "sessions": [
            {
                "token_prefix": s.session_token[:8] + "...",
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat(),
                "ip_address": s.ip_address,
                "is_current": s.session_token == request.headers.get("X-Session-Token"),
            }
            for s in sessions
        ],
        "total": sessions.count(),
    })
""",

    "validators.py": """\
\"\"\"
app/validators.py - Input validation utilities for authentication.

Provides token format validation, email validation, and password strength
checking. All validators return (bool, Optional[str]) tuples where the
string is an error message on failure.
\"\"\"

import re
from typing import Tuple, Optional
from datetime import datetime


# Token format pattern: base64url-safe, at least 48 chars
TOKEN_PATTERN = re.compile(r'^[A-Za-z0-9_-]{48,}$')

# Email pattern (simplified RFC 5322)
EMAIL_PATTERN = re.compile(
    r'^[a-zA-Z0-9.!#$%&\\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}'
    r'[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
)

# Password strength requirements
MIN_PASSWORD_LENGTH = 12
PASSWORD_PATTERNS = {
    "lowercase": re.compile(r'[a-z]'),
    "uppercase": re.compile(r'[A-Z]'),
    "digit": re.compile(r'\\d'),
    "special": re.compile(r'[!@#$%^&*()_+\\-=\\[\\]{};:\\'",.<>?/\\\\|`]'),
}


class AuthenticationError(Exception):
    \"\"\"Raised when authentication fails.\"\"\"
    pass


class TokenValidationError(Exception):
    \"\"\"Raised when a session token fails validation.\"\"\"
    def __init__(self, token: str, reason: str):
        self.token_prefix = token[:8] if len(token) >= 8 else token
        self.reason = reason
        super().__init__(f"Token validation failed: {reason}")


def validate_token_format(token: str) -> Tuple[bool, Optional[str]]:
    \"\"\"Validate that a session token matches the expected format.

    Args:
        token: The session token string to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None on success.
    \"\"\"
    if not token:
        return False, "Token is empty or None"

    if not isinstance(token, str):
        return False, f"Token must be a string, got {type(token).__name__}"

    if len(token) < 16:
        return False, f"Token too short ({len(token)} chars, minimum 16)"

    if len(token) > 512:
        return False, f"Token too long ({len(token)} chars, maximum 512)"

    if not TOKEN_PATTERN.match(token):
        return False, "Token contains invalid characters or does not match expected format"

    return True, None


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    \"\"\"Validate an email address format.

    Args:
        email: The email address to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None on success.
    \"\"\"
    if not email:
        return False, "Email is empty or None"

    if not isinstance(email, str):
        return False, f"Email must be a string, got {type(email).__name__}"

    email = email.strip()

    if len(email) > 254:
        return False, f"Email too long ({len(email)} chars, maximum 254)"

    if not EMAIL_PATTERN.match(email):
        return False, "Email format is invalid"

    # Check for common typos
    local_part = email.split("@")[0]
    if local_part.startswith(".") or local_part.endswith("."):
        return False, "Local part of email cannot start or end with a dot"

    return True, None


def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
    \"\"\"Validate that a password meets strength requirements.

    Requirements:
    - At least 12 characters long
    - Contains at least one lowercase letter
    - Contains at least one uppercase letter
    - Contains at least one digit
    - Contains at least one special character

    Args:
        password: The password to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None on success.
    \"\"\"
    if not password:
        return False, "Password is empty or None"

    if not isinstance(password, str):
        return False, f"Password must be a string, got {type(password).__name__}"

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, (
            f"Password too short ({len(password)} chars, "
            f"minimum {MIN_PASSWORD_LENGTH})"
        )

    missing = []
    if not PASSWORD_PATTERNS["lowercase"].search(password):
        missing.append("lowercase letter")
    if not PASSWORD_PATTERNS["uppercase"].search(password):
        missing.append("uppercase letter")
    if not PASSWORD_PATTERNS["digit"].search(password):
        missing.append("digit")
    if not PASSWORD_PATTERNS["special"].search(password):
        missing.append("special character")

    if missing:
        return False, f"Password missing: {', '.join(missing)}"

    # Check for common passwords
    common = [
        "password1234!", "Admin123456!", "Qwerty123456!",
        "Letmein12345!", "Welcome12345!",
    ]
    if password in common:
        return False, "Password is too common"

    return True, None


def validate_session_expiry(expires_at_str: str) -> Tuple[bool, Optional[str]]:
    \"\"\"Validate a session expiry timestamp string.

    Args:
        expires_at_str: ISO 8601 timestamp string.

    Returns:
        Tuple of (is_valid, error_message). error_message is None on success.
    \"\"\"
    if not expires_at_str:
        return False, "Expiry timestamp is empty"

    try:
        expires_at = datetime.fromisoformat(expires_at_str)
    except (ValueError, TypeError):
        return False, f"Invalid timestamp format: {expires_at_str}"

    if expires_at < datetime.now():
        return False, "Expiry timestamp is in the past"

    return True, None
""",

    "tests.py": """\
\"\"\"
app/tests/test_session.py - Unit and integration tests for session management.

Tests cover session creation, validation, token format checking, and the
race condition scenarios that can occur with concurrent session access.
\"\"\"

import time
import threading
from datetime import timedelta
from unittest.mock import patch, MagicMock

from django.test import TestCase, TransactionTestCase
from django.utils import timezone
from django.core.exceptions import ValidationError

from app.models import User, Session, Role
from app.validators import (
    validate_token_format,
    validate_email,
    validate_password_strength,
    TokenValidationError,
)


class TestTokenFormatValidation(TestCase):
    \"\"\"Tests for the validate_token_format function.\"\"\"

    def test_valid_token(self):
        token = "A" * 48
        is_valid, error = validate_token_format(token)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_empty_token(self):
        is_valid, error = validate_token_format("")
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

    def test_none_token(self):
        is_valid, error = validate_token_format(None)
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

    def test_short_token(self):
        token = "abc123"
        is_valid, error = validate_token_format(token)
        self.assertFalse(is_valid)
        self.assertIn("short", error.lower())

    def test_token_with_special_chars(self):
        token = "a" * 40 + "!@#$"
        is_valid, error = validate_token_format(token)
        self.assertFalse(is_valid)
        self.assertIn("invalid characters", error.lower())

    def test_very_long_token(self):
        token = "a" * 600
        is_valid, error = validate_token_format(token)
        self.assertFalse(is_valid)
        self.assertIn("too long", error.lower())


class TestSessionModel(TestCase):
    \"\"\"Tests for the Session model.\"\"\"

    def setUp(self):
        self.role = Role.objects.create(name="user", permissions={"grants": ["read"]})
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="Str0ngP@ss1234!",
        )
        self.user.role = self.role
        self.user.save()

    def test_create_session(self):
        session = Session.create_for_user(self.user)
        self.assertIsNotNone(session)
        self.assertTrue(session.is_active)
        self.assertTrue(len(session.session_token) >= 48)

    def test_session_expiry(self):
        session = Session.create_for_user(self.user, ttl=timedelta(seconds=1))
        self.assertFalse(session.is_expired())
        # Fast-forward time would show expiry
        # session.expires_at = timezone.now() - timedelta(seconds=1)
        # self.assertTrue(session.is_expired())

    def test_deactivate_session(self):
        session = Session.create_for_user(self.user)
        self.assertTrue(session.is_active)
        session.deactivate()
        session.refresh_from_db()
        self.assertFalse(session.is_active)

    def test_max_active_sessions(self):
        for i in range(User.MAX_ACTIVE_SESSIONS):
            Session.create_for_user(self.user)
        with self.assertRaises(ValidationError):
            Session.create_for_user(self.user)


class TestSessionValidationRaceCondition(TransactionTestCase):
    \"\"\"Tests for the race condition in session token validation.

    BUG: Session.validate_session_token() has a TOCTOU race condition.
    The is_active check and the expiry check are not atomic, so a
    concurrent request can see inconsistent state.
    \"\"\"

    def setUp(self):
        self.role = Role.objects.create(name="user", permissions={"grants": ["read"]})
        self.user = User.objects.create_user(
            username="racetest",
            email="race@example.com",
            password="Str0ngP@ss1234!",
        )
        self.user.role = self.role
        self.user.save()
        self.session = Session.create_for_user(self.user)

    def test_concurrent_validation_and_logout(self):
        \"\"\"Test that concurrent validation and logout do not cause 403 errors.

        This test exposes the race condition: one thread validates the token
        while another thread deactivates the same session. The validation
        should either succeed (pre-deactivation) or fail cleanly, but should
        never return a stale or inconsistent result.
        \"\"\"
        results = []
        errors = []

        def validate_loop():
            for _ in range(50):
                try:
                    result = Session.validate_session_token(
                        self.session.session_token
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

        def logout_loop():
            for _ in range(10):
                try:
                    self.session.deactivate()
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(str(e))

        t_validate = threading.Thread(target=validate_loop)
        t_logout = threading.Thread(target=logout_loop)

        t_validate.start()
        t_logout.start()
        t_validate.join()
        t_logout.join()

        # No exceptions should occur
        self.assertEqual(len(errors), 0, f"Errors during concurrent access: {errors}")

    def test_concurrent_session_creation(self):
        \"\"\"Test that concurrent session creation respects the max limit.\"\"\"
        results = []

        def create_session():
            try:
                session = Session.create_for_user(self.user)
                results.append(("created", session.session_token))
            except ValidationError as e:
                results.append(("rejected", str(e)))

        threads = [threading.Thread(target=create_session) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        created = [r for r in results if r[0] == "created"]
        # Should not exceed max sessions + 1 (the one from setUp)
        self.assertLessEqual(
            len(created), User.MAX_ACTIVE_SESSIONS,
            f"Created {len(created)} sessions, exceeding limit of {User.MAX_ACTIVE_SESSIONS}"
        )


class TestPasswordValidation(TestCase):
    \"\"\"Tests for password strength validation.\"\"\"

    def test_strong_password(self):
        is_valid, error = validate_password_strength("MyStr0ng!P@ssw0rd")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_short_password(self):
        is_valid, error = validate_password_strength("Sh0rt!")
        self.assertFalse(is_valid)
        self.assertIn("short", error.lower())

    def test_no_uppercase(self):
        is_valid, error = validate_password_strength("alllowercase123!")
        self.assertFalse(is_valid)
        self.assertIn("uppercase", error.lower())

    def test_no_special_char(self):
        is_valid, error = validate_password_strength("NoSpecialChar123")
        self.assertFalse(is_valid)
        self.assertIn("special", error.lower())

    def test_common_password(self):
        is_valid, error = validate_password_strength("password1234!")
        self.assertFalse(is_valid)
        self.assertIn("common", error.lower())


class TestEmailValidation(TestCase):
    \"\"\"Tests for email format validation.\"\"\"

    def test_valid_email(self):
        is_valid, error = validate_email("user@example.com")
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    def test_invalid_format(self):
        is_valid, error = validate_email("not-an-email")
        self.assertFalse(is_valid)
        self.assertIn("invalid", error.lower())

    def test_dotted_local(self):
        is_valid, error = validate_email(".user@example.com")
        self.assertFalse(is_valid)
        self.assertIn("dot", error.lower())
""",

    "config.py": """\
\"\"\"
config/settings.py - Application configuration and settings.

Uses environment variables with sensible defaults for all configuration.
Settings are validated at import time to fail fast on misconfiguration.
\"\"\"

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Core settings
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("APP_DEBUG", "false").lower() in ("true", "1", "yes")
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "change-me-in-production-!!")

ALLOWED_HOSTS = os.environ.get("APP_ALLOWED_HOSTS", "*").split(",")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": os.environ.get("DB_ENGINE", "django.db.backends.postgresql"),
        "NAME": os.environ.get("DB_NAME", "auth_db"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
        "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "60")),
        "OPTIONS": {
            "connect_timeout": int(os.environ.get("DB_CONNECT_TIMEOUT", "5")),
        },
    }
}

# ---------------------------------------------------------------------------
# Session settings
# ---------------------------------------------------------------------------
SESSION_TTL_HOURS = int(os.environ.get("SESSION_TTL_HOURS", "24"))
SESSION_MAX_ACTIVE = int(os.environ.get("SESSION_MAX_ACTIVE", "5"))
SESSION_TOKEN_LENGTH = int(os.environ.get("SESSION_TOKEN_LENGTH", "48"))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
CORS_ALLOWED_ORIGINS = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS", "http://localhost:3000"
).split(",")

RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "10"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get(
    "LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": LOG_FORMAT},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "app": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        "django": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}
""",
}

SIMULATED_TEST_OUTPUT_PASS = """\
$ python -m pytest app/tests/test_session.py -v
======================== test session starts ========================
platform linux -- Python 3.11.8, pytest-8.1.1, pluggy-1.4.0
rootdir: /home/user/auth-service
configfile: pyproject.toml
plugins: django-4.8.0, cov-5.0.0
collected 18 items

app/tests/test_session.py::TestTokenFormatValidation::test_valid_token PASSED  [  5%]
app/tests/test_session.py::TestTokenFormatValidation::test_empty_token PASSED   [ 11%]
app/tests/test_session.py::TestTokenFormatValidation::test_none_token PASSED    [ 16%]
app/tests/test_session.py::TestTokenFormatValidation::test_short_token PASSED   [ 22%]
app/tests/test_session.py::TestTokenFormatValidation::test_token_with_special_chars PASSED [ 27%]
app/tests/test_session.py::TestTokenFormatValidation::test_very_long_token PASSED [ 33%]
app/tests/test_session.py::TestSessionModel::test_create_session PASSED         [ 38%]
app/tests/test_session.py::TestSessionModel::test_session_expiry PASSED          [ 44%]
app/tests/test_session.py::TestSessionModel::test_deactivate_session PASSED     [ 50%]
app/tests/test_session.py::TestSessionModel::test_max_active_sessions PASSED    [ 55%]
app/tests/test_session.py::TestSessionValidationRaceCondition::test_concurrent_validation_and_logout PASSED [ 61%]
app/tests/test_session.py::TestSessionValidationRaceCondition::test_concurrent_session_creation PASSED [ 66%]
app/tests/test_session.py::TestPasswordValidation::test_strong_password PASSED   [ 72%]
app/tests/test_session.py::TestPasswordValidation::test_short_password PASSED    [ 77%]
app/tests/test_session.py::TestPasswordValidation::test_no_uppercase PASSED      [ 83%]
app/tests/test_session.py::TestPasswordValidation::test_no_special_char PASSED   [ 88%]
app/tests/test_session.py::TestEmailValidation::test_valid_email PASSED          [ 94%]
app/tests/test_session.py::TestEmailValidation::test_invalid_format PASSED       [100%]

======================== 18 passed in 2.34s ========================
"""

SIMULATED_TEST_OUTPUT_FAIL = """\
$ python -m pytest app/tests/test_session.py -v
======================== test session starts ========================
platform linux -- Python 3.11.8, pytest-8.1.1, pluggy-1.4.0
rootdir: /home/user/auth-service
configfile: pyproject.toml
plugins: django-4.8.0, cov-5.0.0
collected 18 items

app/tests/test_session.py::TestTokenFormatValidation::test_valid_token PASSED  [  5%]
app/tests/test_session.py::TestTokenFormatValidation::test_empty_token PASSED   [ 11%]
app/tests/test_session.py::TestTokenFormatValidation::test_none_token PASSED    [ 16%]
app/tests/test_session.py::TestTokenFormatValidation::test_short_token PASSED   [ 22%]
app/tests/test_session.py::TestTokenFormatValidation::test_token_with_special_chars PASSED [ 27%]
app/tests/test_session.py::TestTokenFormatValidation::test_very_long_token PASSED [ 33%]
app/tests/test_session.py::TestSessionModel::test_create_session PASSED         [ 38%]
app/tests/test_session.py::TestSessionModel::test_session_expiry PASSED          [ 44%]
app/tests/test_session.py::TestSessionModel::test_deactivate_session PASSED     [ 50%]
app/tests/test_session.py::TestSessionModel::test_max_active_sessions PASSED    [ 55%]
app/tests/test_session.py::TestSessionValidationRaceCondition::test_concurrent_validation_and_logout FAILED [61%]
app/tests/test_session.py::TestSessionValidationRaceCondition::test_concurrent_session_creation PASSED [ 66%]
app/tests/test_session.py::TestPasswordValidation::test_strong_password PASSED   [ 72%]
app/tests/test_session.py::TestPasswordValidation::test_short_password PASSED    [ 77%]
app/tests/test_session.py::TestPasswordValidation::test_no_uppercase PASSED      [ 83%]
app/tests/test_session.py::TestPasswordValidation::test_no_special_char PASSED   [ 88%]
app/tests/test_session.py::TestEmailValidation::test_valid_email PASSED          [ 94%]
app/tests/test_session.py::TestEmailValidation::test_invalid_format PASSED       [100%]

============================= FAILURES ==============================
____ TestSessionValidationRaceCondition::test_concurrent_validation_and_logout ____

self = <app.tests.test_session.TestSessionValidationRaceCondition testMethod=test_concurrent_validation_and_logout>

    def test_concurrent_validation_and_logout(self):
        \"\"\"Test that concurrent validation and logout do not cause 403 errors.\"\"\"
        results = []
        errors = []

        def validate_loop():
            for _ in range(50):
                try:
                    result = Session.validate_session_token(
                        self.session.session_token
                    )
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))

        def logout_loop():
            for _ in range(10):
                try:
                    self.session.deactivate()
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(str(e))

        t_validate = threading.Thread(target=validate_loop)
        t_logout = threading.Thread(target=logout_loop)

        t_validate.start()
        t_logout.start()
        t_validate.join()
        t_logout.join()

>       self.assertEqual(len(errors), 0, f"Errors during concurrent access: {errors}")
E       AssertionError: 3 != 0 : Errors during concurrent access: [
            'Session matching query does not exist.',
            'Session matching query does not exist.',
            'Session matching query does not exist.'
          ]

app/tests/test_session.py:142: AssertionError
===================== 1 failed, 17 passed in 2.87s ======================
"""


# Ordered list of file keys for sequential selection in LOCATE stage
FILE_ORDER = ["models.py", "views.py", "validators.py", "tests.py", "config.py"]


def get_file_content_for_locate(locate_call_idx: int) -> str:
    """Select file content for the Nth LOCATE call, rotating through files.

    Unlike the old _pick_file_for_locate which randomly selected (causing
    repeated identical content when same stage repeats), this rotates
    sequentially so each LOCATE call gets genuinely different content.

    Args:
        locate_call_idx: 0-based index of this LOCATE call within the job.

    Returns:
        Simulated file read output for injection as tool result.
    """
    primary_key = FILE_ORDER[locate_call_idx % len(FILE_ORDER)]
    content = SIMULATED_FILE_CONTENTS[primary_key]
    return f'--- read_file("app/{primary_key}") ---\n{content}'


def get_test_result_for_verify(verify_call_idx: int) -> str:
    """Select test result for the Nth VERIFY call.

    Strategy:
    - First verify (verify_call_idx=0): PASS (initial implementation works)
    - Second verify (first debug loop, verify_call_idx=1): FAIL (bug found)
    - Third verify (verify_call_idx=2): PASS (bug fixed)
    - Alternates thereafter

    This creates a realistic narrative: initial fix seems to work,
    but testing reveals edge cases, which are then fixed.

    Args:
        verify_call_idx: 0-based index of this VERIFY call within the job.

    Returns:
        Simulated test output string (pass or fail).
    """
    if verify_call_idx == 0:
        return SIMULATED_TEST_OUTPUT_PASS
    elif verify_call_idx % 2 == 1:
        return SIMULATED_TEST_OUTPUT_FAIL
    else:
        return SIMULATED_TEST_OUTPUT_PASS
