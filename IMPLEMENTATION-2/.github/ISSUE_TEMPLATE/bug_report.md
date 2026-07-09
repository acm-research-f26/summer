name: Bug Report
description: Report a bug in SleepEEGPy
labels: ["bug"]

body:
  - type: markdown
    attributes:
      value: |
        Thank you for reporting a bug! Please provide as much detail as possible.

  - type: textarea
    attributes:
      label: Description
      description: A clear and concise description of the bug.
      placeholder: "Describe the issue here..."
    validations:
      required: true

  - type: textarea
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior.
      placeholder: |
        1. Load data using...
        2. Run preprocessing with...
        3. Error occurs when...
    validations:
      required: true

  - type: textarea
    attributes:
      label: Expected Behavior
      description: Describe what you expected to happen.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Error Message
      description: Include the full error traceback if applicable.
      render: python

  - type: input
    attributes:
      label: Python Version
      description: Output of `python --version`
      placeholder: "3.9.0"

  - type: input
    attributes:
      label: SleepEEGPy Version
      description: Version of SleepEEGPy you're using
      placeholder: "0.1.0"

  - type: textarea
    attributes:
      label: Additional Context
      description: Any other context about the problem.
