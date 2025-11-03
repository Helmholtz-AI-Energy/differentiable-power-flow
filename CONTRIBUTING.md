# Contributing to differentiable-power-flow

Welcome to ``differentiable-power-flow`` :sunglasses:! We're thrilled that you're interested in contributing to our open-source project.
By participating, you can help improve the project and make it even better.

## How to Contribute

1. **Fork the Repository**: Click the "Fork" button at the top right corner of this repository's page to create your own copy.

2. **Clone Your Fork**: Clone your forked repository to your local machine using Git :octocat::
   ```bash
   git clone git@github.com:Helmholtz-AI-Energy/differentiable-power-flow.git
   ```

3. **Create a Branch**: Create a new branch for your contribution. Choose a descriptive name. Depending on what you want
   to work on, prepend either of the following prefixes, `features`, `maintenance`, `bugfix`, or `hotfix`. Example:
   ```bash
   git checkout -b features/your-feature-name
   ```

4. **Commit Changes**: Commit your changes with a clear and concise commit message:
   ```bash
   git commit -m "Add your commit message here"
   ```

5. **Push Changes**: Push your changes to your fork on GitHub:
   ```bash
   git push origin your-feature-name
   ```

6. **Rebase Onto Current Main:** Rebase your feature branch onto the current main branch of the original repo. Leaving
   this step out might lead to problems with the test workflow when merging your branch into the main later on. Rebasing
   will include any changes that might have been pushed into the main in the meantime and resolve possible conflicts.
   To sync your fork with the original upstream repo, check out [this page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
   or follow the steps below. Note that before you can sync your fork with an upstream repo, you must configure a remote
   that points to the upstream repository in Git.
   ```
   cd <path/to/your/local/project/fork>
   git fetch upstream
   git checkout main
   git merge upstream/main
   git rebase main features/your-feature-name
   ```

7. **Open a Pull Request**: Go to the [original repository](https://github.com/Helmholtz-AI-Energy/differentiable-power-flow) and click the "New Pull Request" button. Follow the guidelines in the template to submit your pull request.

## Code of Conduct

Please note that we have a [Code of Conduct](CODE_OF_CONDUCT.md), and we expect all contributors to follow it. Be kind and respectful to one another.

## Questions or Issues

If you have questions or encounter any issues, please create an issue in the [Issues](https://github.com/Helmholtz-AI-Energy/differentiable-power-flow/issues) section of this repository.

Thank you for your contribution!
