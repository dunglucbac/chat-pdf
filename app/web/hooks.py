import functools
import tempfile
import uuid
import os
import logging
from flask import g, session, request
from sqlalchemy.exc import IntegrityError, NoResultFound
from werkzeug.exceptions import Unauthorized, BadRequest
from app.web.db.models import User, Model


def load_model(Model: Model, extract_id_lambda=None):
    """
    Decorator factory that loads a SQLAlchemy model instance from the database
    and injects it into the view function, replacing the raw ID argument.

    Also enforces ownership — raises Unauthorized if the loaded instance does
    not belong to the currently logged-in user (g.user).

    Args:
        Model (Model): The SQLAlchemy model class to query (e.g. Conversation, Pdf).
        extract_id_lambda (Callable[[Request], str] | None): Optional function that
            receives the Flask request and returns the model ID. Use this when the ID
            comes from a query string parameter (e.g. ``lambda r: r.args.get("pdf_id")``).
            If omitted, the ID is read from the URL path kwargs using the convention
            ``{model_name}_id`` (e.g. ``conversation_id`` for Conversation).

    Returns:
        Callable: A decorator that wraps a view function with fetch + auth logic.

    Raises:
        ValueError: If the model ID cannot be found in either kwargs or the lambda.
        NoResultFound: If no record with the given ID exists in the database (→ 404).
        Unauthorized: If the record's user_id does not match g.user.id (→ 401).

    Example:
        @bp.route("/<string:conversation_id>/messages", methods=["POST"])
        @login_required
        @load_model(Conversation)
        def create_message(conversation):
            ...

        @bp.route("/", methods=["POST"])
        @login_required
        @load_model(Pdf, lambda r: r.args.get("pdf_id"))
        def create_conversation(pdf):
            ...
    """
    def decorator(view):
        @functools.wraps(view)
        def wrapped_view(**kwargs):
            model_name = Model.__name__.lower()
            model_id_name = f"{model_name}_id"

            model_id = kwargs.get(model_id_name)
            if extract_id_lambda:
                model_id = extract_id_lambda(request)

            if not model_id:
                raise ValueError(f"{model_id_name} must be provided in the request.")

            instance = Model.find_by(id=model_id)

            if instance.user_id != g.user.id:
                raise Unauthorized("You are not authorized to view this.")

            if model_id_name in kwargs:
                del kwargs[model_id_name]
            kwargs[model_name] = instance
            return view(**kwargs)

        return wrapped_view

    return decorator


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return {"message": "Unauthorized"}, 401
        return view(**kwargs)

    return wrapped_view


def add_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


def load_logged_in_user():
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
    else:
        try:
            g.user = User.find_by(id=user_id)
        except Exception:
            g.user = None


def handle_file_upload(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        file = request.files["file"]
        file_id = str(uuid.uuid4())

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file_id)
            file.save(file_path)

            kwargs["file_id"] = file_id
            kwargs["file_path"] = file_path
            kwargs["file_name"] = file.filename
            return fn(*args, **kwargs)

    return wrapped


def handle_error(err):
    if isinstance(err, IntegrityError):
        logging.error(err)
        return {"message": "In use"}, 400
    elif isinstance(err, NoResultFound):
        logging.error(err)
        return {"message": "Not found"}, 404
    elif isinstance(err, Unauthorized):
        logging.error(err)
        return {"message": err.description}, 401
    elif isinstance(err, BadRequest):
        logging.error(err)
        return {"message": err.description}, 401

    raise err
