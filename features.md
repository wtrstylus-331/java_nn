This is mainly a document to keep track of current branches meant to implement certain aspects, or to simply refactor code to be more efficient/modular.

# Branches
### master
- Contains all up-to-date code

### Optimizers
- Similarly to the architecture for implementing logic for activations, have the architecture defined for optimizers for now such as GD (stochastic, full/mini batch) and Adam

# Past Branches
### ActivationRefactor
#### Merged 23/11/2026
- Have a super parent interface and allow creation of activation functions as subclass that implement, and override respective logic
