"""Vertex AI CustomTrainingJob entrypoint (runs as aiplatform_custom_trainer_script.task)."""

try:
    from .vertex_entrypoint import (
        logger,
        parse_args,
        resolve_model_dir,
        resolve_training_dir,
        train,
    )
except ImportError:
    from vertex_entrypoint import (
        logger,
        parse_args,
        resolve_model_dir,
        resolve_training_dir,
        train,
    )

if __name__ == "__main__":
    import sys

    args = parse_args()
    try:
        train(
            training_dir=resolve_training_dir(args.training),
            model_dir=resolve_model_dir(args.model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception("Vertex AI entrypoint failed")
        sys.exit(1)
