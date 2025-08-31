# Contributing to `ossm-tasks`

The `ossm-tasks` package implements the **Sensorimotor Task and Environment Format (STEF)** and provides a catalogue of standardized tasks.

## How to Contribute
- **Issues**: Use GitHub Issues to report bugs, request features, or propose new tasks.  
- **Pull Requests**: Fork the repo, create a feature branch, and open a PR. Keep PRs focused and provide a clear description.  
- **Discussions**: Use GitHub Discussions for broader design proposals or standards questions.

## Adding a Task
When contributing a new task:
1. Define the task in an XML file following [`STEF.xsd`](./STEF.xsd).  
2. Place the XML in the appropriate `suite/` folder.  
3. Add minimal tests or usage examples.  
4. Document the task with a short description (inputs, outputs, objectives).  

## Code Style
- Follow **PEP 8** for Python code (where applicable).  
- Keep task definitions well-structured and schema-compliant.  

## Tests & Validation
- Run validation against `STEF.xsd` for every contributed task.  
- Ensure tests pass before submitting a PR.  

## Licensing
All contributions are released under **GPL-3.0**. By contributing, you agree to this license.

## Code of Conduct
All contributors must follow the project’s [Code of Conduct](../CODE_OF_CONDUCT.md).
