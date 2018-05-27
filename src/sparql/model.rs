use model::data::*;
use std::fmt;
use uuid::Uuid;

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Hash)]
pub struct Variable {
    name: String,
}

impl Variable {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "?{}", self.name)
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self {
            name: Uuid::new_v4().to_string(),
        }
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Hash)]
pub enum NamedNodeOrVariable {
    NamedNode(NamedNode),
    Variable(Variable),
}

impl fmt::Display for NamedNodeOrVariable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            NamedNodeOrVariable::NamedNode(node) => write!(f, "{}", node),
            NamedNodeOrVariable::Variable(var) => write!(f, "{}", var),
        }
    }
}

impl From<NamedNode> for NamedNodeOrVariable {
    fn from(node: NamedNode) -> Self {
        NamedNodeOrVariable::NamedNode(node)
    }
}

impl From<Variable> for NamedNodeOrVariable {
    fn from(var: Variable) -> Self {
        NamedNodeOrVariable::Variable(var)
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Hash)]
pub enum TermOrVariable {
    Term(Term),
    Variable(Variable),
}

impl fmt::Display for TermOrVariable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TermOrVariable::Term(node) => write!(f, "{}", node),
            TermOrVariable::Variable(var) => write!(f, "{}", var),
        }
    }
}

impl From<NamedNode> for TermOrVariable {
    fn from(node: NamedNode) -> Self {
        TermOrVariable::Term(node.into())
    }
}

impl From<BlankNode> for TermOrVariable {
    fn from(node: BlankNode) -> Self {
        TermOrVariable::Term(node.into())
    }
}

impl From<Literal> for TermOrVariable {
    fn from(literal: Literal) -> Self {
        TermOrVariable::Term(literal.into())
    }
}

impl From<NamedOrBlankNode> for TermOrVariable {
    fn from(node: NamedOrBlankNode) -> Self {
        TermOrVariable::Term(node.into())
    }
}

impl From<Term> for TermOrVariable {
    fn from(node: Term) -> Self {
        TermOrVariable::Term(node)
    }
}

impl From<Variable> for TermOrVariable {
    fn from(var: Variable) -> Self {
        TermOrVariable::Variable(var)
    }
}

impl From<NamedNodeOrVariable> for TermOrVariable {
    fn from(element: NamedNodeOrVariable) -> Self {
        match element {
            NamedNodeOrVariable::NamedNode(node) => TermOrVariable::Term(node.into()),
            NamedNodeOrVariable::Variable(var) => TermOrVariable::Variable(var),
        }
    }
}
