name: Feature Request
description: Suggest a new feature or improvement
labels: ["enhancement"]

body:
  - type: markdown
    attributes:
      value: |
        Thank you for the feature request! Please describe the feature you'd like to see.

  - type: textarea
    attributes:
      label: Description
      description: Clear and concise description of the feature.
      placeholder: "I would like to see..."
    validations:
      required: true

  - type: textarea
    attributes:
      label: Use Case
      description: Describe the use case or problem this feature would solve.
    validations:
      required: true

  - type: textarea
    attributes:
      label: Proposed Implementation
      description: Optional description of how you think this feature should be implemented.

  - type: textarea
    attributes:
      label: Additional Context
      description: Any other context or examples.
