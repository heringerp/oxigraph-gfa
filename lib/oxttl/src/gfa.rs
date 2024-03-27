//! A [GFA](https://www.w3.org/TR/turtle/) streaming parser implemented by [`GFAParser`]
//! and a serializer implemented by [`GFASerializer`].

#[cfg(feature = "async-tokio")]
use crate::toolkit::FromTokioAsyncReadIterator;
use crate::toolkit::ParseError;
#[cfg(feature = "async-tokio")]
use crate::trig::ToTokioAsyncWriteTriGWriter;
use crate::trig::{LowLevelTriGWriter, ToWriteTriGWriter, TriGSerializer};
use gfa::parser::GFAParser as HandlegraphGFAParser;
use handlegraph::conversion::from_gfa;
use handlegraph::handlegraph::{IntoHandles, IntoSequences, IntoEdges};
use handlegraph::handle::{Handle, Edge};
use handlegraph::packedgraph::PackedGraph;
use handlegraph::packedgraph::paths::StepPtr;
use handlegraph::pathhandlegraph::{PathId, IntoPathIds, GraphPathNames, GraphPaths};
use oxiri::{Iri, IriParseError};
use oxrdf::vocab::{rdf, vg};
use oxrdf::{GraphNameRef, Triple, TripleRef, NamedNode, Literal};
use std::collections::hash_map::Iter;
use std::collections::HashMap;
use std::io::{self, Read, Write, Error, ErrorKind};
use std::vec::IntoIter;

const FIRST_RANK: u64 = 1;

#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, AsyncWrite};

/// A [GFA](https://www.w3.org/TR/turtle/) streaming parser.
///
/// Support for [GFA-star](https://w3c.github.io/rdf-star/cg-spec/2021-12-17.html#turtle-star) is available behind the `rdf-star` feature and the [`GFAParser::with_quoted_triples`] option.
///
/// Count the number of people:
/// ```
/// use oxrdf::vocab::rdf;
/// use oxrdf::NamedNodeRef;
/// use oxttl::GFAParser;
///
/// let file = br#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for triple in GFAParser::new().parse_read(file.as_ref()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default)]
#[must_use]
pub struct GFAParser {
    unchecked: bool,
    base: Option<Iri<String>>,
    prefixes: HashMap<String, Iri<String>>,
    #[cfg(feature = "rdf-star")]
    with_quoted_triples: bool,
}

impl GFAParser {
    /// Builds a new [`GFAParser`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assumes the file is valid to make parsing faster.
    ///
    /// It will skip some validations.
    ///
    /// Note that if the file is actually not valid, then broken RDF might be emitted by the parser.
    #[inline]
    pub fn unchecked(mut self) -> Self {
        self.unchecked = true;
        self
    }

    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        self.prefixes
            .insert(prefix_name.into(), Iri::parse(prefix_iri.into())?);
        Ok(self)
    }

    /// Enables [GFA-star](https://w3c.github.io/rdf-star/cg-spec/2021-12-17.html#turtle-star).
    #[cfg(feature = "rdf-star")]
    #[inline]
    pub fn with_quoted_triples(mut self) -> Self {
        self.with_quoted_triples = true;
        self
    }

    /// Parses a GFA file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxrdf::vocab::rdf;
    /// use oxrdf::NamedNodeRef;
    /// use oxttl::GFAParser;
    ///
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" .
    /// <bar> a schema:Person ;
    ///     schema:name "Bar" ."#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for triple in GFAParser::new().parse_read(file.as_ref()) {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn parse_read<R: Read>(self, read: R) -> FromReadGFAReader {
        FromReadGFAReader {
            inner: self.parse(read),
        }
    }

    /// Parses a GFA file from a [`AsyncRead`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxrdf::vocab::rdf;
    /// use oxrdf::NamedNodeRef;
    /// use oxttl::GFAParser;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), oxttl::ParseError> {
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" .
    /// <bar> a schema:Person ;
    ///     schema:name "Bar" ."#;
    ///
    /// let schema_person = NamedNodeRef::new_unchecked("http://schema.org/Person");
    /// let mut count = 0;
    /// let mut parser = GFAParser::new().parse_tokio_async_read(file.as_ref());
    /// while let Some(triple) = parser.next().await {
    ///     let triple = triple?;
    ///     if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-tokio")]
    pub fn parse_tokio_async_read<R: AsyncRead + Unpin>(
        self,
        read: R,
    ) -> FromTokioAsyncReadGFAReader<R> {
        FromTokioAsyncReadGFAReader {
            inner: self.parse().parser.parse_tokio_async_read(read),
        }
    }

    // /// Allows to parse a GFA file by using a low-level API.
    // ///
    // /// Count the number of people:
    // /// ```
    // /// use oxrdf::vocab::rdf;
    // /// use oxrdf::NamedNodeRef;
    // /// use oxttl::GFAParser;
    // ///
    // /// let file: [&[u8]; 5] = [
    // ///     b"@base <http://example.com/>",
    // ///     b". @prefix schema: <http://schema.org/> .",
    // ///     b"<foo> a schema:Person",
    // ///     b" ; schema:name \"Foo\" . <bar>",
    // ///     b" a schema:Person ; schema:name \"Bar\" .",
    // /// ];
    // ///
    // /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    // /// let mut count = 0;
    // /// let mut parser = GFAParser::new().parse();
    // /// let mut file_chunks = file.iter();
    // /// while !parser.is_end() {
    // ///     // We feed more data to the parser
    // ///     if let Some(chunk) = file_chunks.next() {
    // ///         parser.extend_from_slice(chunk);
    // ///     } else {
    // ///         parser.end(); // It's finished
    // ///     }
    // ///     // We read as many triples from the parser as possible
    // ///     while let Some(triple) = parser.read_next() {
    // ///         let triple = triple?;
    // ///         if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
    // ///             count += 1;
    // ///         }
    // ///     }
    // /// }
    // /// assert_eq!(2, count);
    // /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    // /// ```
    pub fn parse<R: Read>(self, read: R) -> LowLevelGFAReader {
        LowLevelGFAReader::new(
            read,
            self.unchecked,
            self.base,
            self.prefixes,
            #[cfg(feature = "rdf-star")]
            self.with_quoted_triples,
        )
    }
}

/// Parses a GFA file from a [`Read`] implementation. Can be built using [`GFAParser::parse_read`].
///
/// Count the number of people:
/// ```
/// use oxrdf::vocab::rdf;
/// use oxrdf::NamedNodeRef;
/// use oxttl::GFAParser;
///
/// let file = br#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for triple in GFAParser::new().parse_read(file.as_ref()) {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct FromReadGFAReader {
    inner: LowLevelGFAReader,
}

impl FromReadGFAReader {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse_read(file.as_ref());
    /// assert!(reader.prefixes().collect::<Vec<_>>().is_empty()); // No prefix at the beginning
    ///
    /// reader.next().unwrap()?; // We read the first triple
    /// assert_eq!(
    ///     reader.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> GFAPrefixesIter<'_> {
        GFAPrefixesIter {
            inner: self.inner.prefixes.iter(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse_read(file.as_ref());
    /// assert!(reader.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// reader.next().unwrap()?; // We read the first triple
    /// assert_eq!(reader.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner
            .base
            .as_ref()
            .map(Iri::as_str)
    }
}

impl Iterator for FromReadGFAReader {
    type Item = Result<Triple, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.inner.next()?.map(Into::into))
    }
}

/// Parses a GFA file from a [`AsyncRead`] implementation. Can be built using [`GFAParser::parse_tokio_async_read`].
///
/// Count the number of people:
/// ```
/// use oxrdf::vocab::rdf;
/// use oxrdf::NamedNodeRef;
/// use oxttl::GFAParser;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), oxttl::ParseError> {
/// let file = br#"@base <http://example.com/> .
/// @prefix schema: <http://schema.org/> .
/// <foo> a schema:Person ;
///     schema:name "Foo" .
/// <bar> a schema:Person ;
///     schema:name "Bar" ."#;
///
/// let schema_person = NamedNodeRef::new_unchecked("http://schema.org/Person");
/// let mut count = 0;
/// let mut parser = GFAParser::new().parse_tokio_async_read(file.as_ref());
/// while let Some(triple) = parser.next().await {
///     let triple = triple?;
///     if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct FromTokioAsyncReadGFAReader<R: AsyncRead + Unpin> {
    inner: FromTokioAsyncReadIterator<R, TriGRecognizer>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> FromTokioAsyncReadGFAReader<R> {
    /// Reads the next triple or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Triple, ParseError>> {
        Some(self.inner.next().await?.map(Into::into))
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), oxttl::ParseError> {
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse_tokio_async_read(file.as_ref());
    /// assert_eq!(reader.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// reader.next().await.unwrap()?; // We read the first triple
    /// assert_eq!(
    ///     reader.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// # Ok(())
    /// # }
    /// ```
    pub fn prefixes(&self) -> GFAPrefixesIter<'_> {
        GFAPrefixesIter {
            inner: self.inner.parser.context.prefixes(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), oxttl::ParseError> {
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse_tokio_async_read(file.as_ref());
    /// assert!(reader.base_iri().is_none()); // No base IRI at the beginning
    ///
    /// reader.next().await.unwrap()?; // We read the first triple
    /// assert_eq!(reader.base_iri(), Some("http://example.com/")); // There is now a base IRI
    /// # Ok(())
    /// # }
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner
            .parser
            .context
            .lexer_options
            .base_iri
            .as_ref()
            .map(Iri::as_str)
    }
}

/// Parses a GFA file by using a low-level API. Can be built using [`GFAParser::parse`].
///
/// Count the number of people:
/// ```
/// use oxrdf::vocab::rdf;
/// use oxrdf::NamedNodeRef;
/// use oxttl::GFAParser;
///
/// let file: [&[u8]; 5] = [
///     b"@base <http://example.com/>",
///     b". @prefix schema: <http://schema.org/> .",
///     b"<foo> a schema:Person",
///     b" ; schema:name \"Foo\" . <bar>",
///     b" a schema:Person ; schema:name \"Bar\" .",
/// ];
///
/// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
/// let mut count = 0;
/// let mut parser = GFAParser::new().parse();
/// let mut file_chunks = file.iter();
/// while !parser.is_end() {
///     // We feed more data to the parser
///     if let Some(chunk) = file_chunks.next() {
///         parser.extend_from_slice(chunk);
///     } else {
///         parser.end(); // It's finished
///     }
///     // We read as many triples from the parser as possible
///     while let Some(triple) = parser.read_next() {
///         let triple = triple?;
///         if triple.predicate == rdf::TYPE && triple.object == schema_person.into() {
///             count += 1;
///         }
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelGFAReader {
    graph: PackedGraph,
    #[allow(dead_code)]
    unchecked: bool,
    base: Option<Iri<String>>,
    prefixes: HashMap<String, Iri<String>>,
    #[allow(dead_code)]
    #[cfg(feature = "rdf-star")]
    with_quoted_triples: bool,
    state: GFAState,
}

impl LowLevelGFAReader {
    pub fn new<R: Read>(mut read: R,
               unchecked: bool,
               base: Option<Iri<String>>,
               prefixes: HashMap<String, Iri<String>>,
               #[cfg(feature = "rdf-star")]
               with_quoted_triples: bool) -> Self {
        let mut text: String = String::new();
        if read.read_to_string(&mut text).is_err() {
            return Self {
                graph: PackedGraph::new(),
                unchecked,
                base,
                prefixes,
                #[cfg(feature = "rdf-star")]
                with_quoted_triples,
                state: GFAState::Invalid,
            }
        }
        let gfa_parser = HandlegraphGFAParser::new();
        if let Ok(gfa) = gfa_parser
            .parse_lines(text.lines().map(|s| s.as_bytes())) {
            let graph = from_gfa::<PackedGraph, ()>(&gfa);
            println!("Finished reading graph from gfa file");
            Self {
                graph,
                unchecked,
                base,
                prefixes,
                #[cfg(feature = "rdf-star")]
                with_quoted_triples,
                state: GFAState::Start,
            }
        } else {
            Self {
                graph: PackedGraph::new(),
                unchecked,
                base,
                prefixes,
                #[cfg(feature = "rdf-star")]
                with_quoted_triples,
                state: GFAState::Invalid,
            }

        }
    }
    pub fn next(&mut self) -> Option<Result<Triple, ParseError>> {
        match &mut self.state {
            GFAState::Finished => {
                println!("Finished all GFA parsing");
                None
            },
            GFAState::Invalid => {
                self.state = GFAState::Finished;
                Some(Err(ParseError::Io(io::Error::new(io::ErrorKind::Other, "Invalid GFA file"))))
            }
            GFAState::Start => {
                let iter = self.graph.handles().collect::<Vec<_>>().into_iter();
                self.state = GFAState::NodeType(iter);
                self.next()
            }
            GFAState::NodeType(iter) => {
                if let Some(handle) = iter.next() {
                    Some(self.get_node_type_triple(handle))
                } else {
                    let iter = self.graph.handles().collect::<Vec<_>>().into_iter();
                    self.state = GFAState::NodeValue(iter);
                    self.next()
                }
            }
            GFAState::NodeValue(iter) => {
                if let Some(handle) = iter.next() {
                    Some(self.get_node_value_triple(handle))
                } else {
                    println!("Finished nodes");
                    let iter = self.graph.edges().collect::<Vec<_>>().into_iter();
                    self.state = GFAState::EdgeLink(iter);
                    self.next()
                }
            }
            GFAState::EdgeLink(iter) => {
                if let Some(edge) = iter.next() {
                    Some(self.get_edge_link_triple(edge))
                } else {
                    let iter = self.graph.edges().collect::<Vec<_>>().into_iter();
                    self.state = GFAState::EdgeLinkDirectional(iter);
                    self.next()
                }
            }
            GFAState::EdgeLinkDirectional(iter) => {
                if let Some(edge) = iter.next() {
                    Some(self.get_edge_link_directional_triple(edge))
                } else {
                    println!("Finished edges");
                    let iter = self.graph.path_ids().collect::<Vec<_>>().into_iter();
                    self.state = GFAState::PathType(iter);
                    self.next()
                }
            }
            GFAState::PathType(iter) => {
                if let Some(path_id) = iter.next() {
                    Some(self.get_path_type_triple(path_id))
                } else {
                    println!("Finished paths");
                    let mut iter = self.graph.path_ids().collect::<Vec<_>>().into_iter();
                    if let Some(first_path) = iter.next() {
                        if let Some(first_step) = self.graph.path_first_step(first_path) {
                            self.state = GFAState::StepType(iter, first_path, first_step, FIRST_RANK);
                        } else {
                            self.state = GFAState::Finished;
                        }
                    } else {
                        self.state = GFAState::Finished;
                    }
                    self.next()
                }
            }
            GFAState::StepType(iter, path_id, step, rank) => {
                let path_id = *path_id;
                let rank = *rank;
                let step = *step;
                let mut iter = iter.to_owned();
                let triple = self.get_step_type_triple(path_id, rank);
                if let Some(next_step) = self.graph.path_next_step(path_id, step) {
                    self.state = GFAState::StepType(iter, path_id, next_step, rank + 1);
                } else if let Some(next_path_id) = iter.next() {
                    if let Some(first_step) = self.graph.path_first_step(next_path_id) {
                        self.state = GFAState::StepType(iter, next_path_id, first_step, FIRST_RANK);
                    } else {
                        self.state = GFAState::Invalid;
                    }
                } else {
                    let mut iter = self.graph.path_ids().collect::<Vec<_>>().into_iter();
                    if let Some(first_path) = iter.next() {
                        if let Some(first_step) = self.graph.path_first_step(first_path) {
                            self.state = GFAState::StepRank(iter, first_path, first_step, FIRST_RANK);
                        } else {
                            self.state = GFAState::Finished;
                        }
                    } else {
                        self.state = GFAState::Finished;
                    }
                }
                Some(triple)
            }
            GFAState::StepRank(iter, path_id, step, rank) => {
                let path_id = *path_id;
                let rank = *rank;
                let step = *step;
                let mut iter = iter.to_owned();
                let triple = self.get_step_rank_triple(path_id, rank);
                if let Some(next_step) = self.graph.path_next_step(path_id, step) {
                    self.state = GFAState::StepRank(iter, path_id, next_step, rank + 1);
                } else if let Some(next_path_id) = iter.next() {
                    if let Some(first_step) = self.graph.path_first_step(next_path_id) {
                        self.state = GFAState::StepRank(iter, next_path_id, first_step, FIRST_RANK);
                    } else {
                        self.state = GFAState::Invalid;
                    }
                } else {
                    let mut iter = self.graph.path_ids().collect::<Vec<_>>().into_iter();
                    if let Some(first_path) = iter.next() {
                        if let Some(first_step) = self.graph.path_first_step(first_path) {
                            self.state = GFAState::StepNode(iter, first_path, first_step, FIRST_RANK);
                        } else {
                            self.state = GFAState::Finished;
                        }
                    } else {
                        self.state = GFAState::Finished;
                    }
                }
                Some(triple)
            }
            GFAState::StepNode(iter, path_id, step, rank) => {
                let path_id = *path_id;
                let rank = *rank;
                let step = *step;
                let mut iter = iter.to_owned();
                let triple = self.get_step_node_triple(path_id, rank, step);
                if let Some(next_step) = self.graph.path_next_step(path_id, step) {
                    self.state = GFAState::StepNode(iter, path_id, next_step, rank + 1);
                } else if let Some(next_path_id) = iter.next() {
                    if let Some(first_step) = self.graph.path_first_step(next_path_id) {
                        self.state = GFAState::StepNode(iter, next_path_id, first_step, FIRST_RANK);
                    } else {
                        self.state = GFAState::Invalid;
                    }
                } else {
                    let mut iter = self.graph.path_ids().collect::<Vec<_>>().into_iter();
                    if let Some(first_path) = iter.next() {
                        if let Some(first_step) = self.graph.path_first_step(first_path) {
                            self.state = GFAState::StepPath(iter, first_path, first_step, FIRST_RANK);
                        } else {
                            self.state = GFAState::Finished;
                        }
                    } else {
                        self.state = GFAState::Finished;
                    }
                }
                Some(triple)
            }
            GFAState::StepPath(iter, path_id, step, rank) => {
                let path_id = *path_id;
                let rank = *rank;
                let step = *step;
                let mut iter = iter.to_owned();
                let triple = self.get_step_path_triple(path_id, rank);
                if let Some(next_step) = self.graph.path_next_step(path_id, step) {
                    self.state = GFAState::StepPath(iter, path_id, next_step, rank + 1);
                } else if let Some(next_path_id) = iter.next() {
                    if let Some(first_step) = self.graph.path_first_step(next_path_id) {
                        self.state = GFAState::StepPath(iter, next_path_id, first_step, FIRST_RANK);
                    } else {
                        self.state = GFAState::Invalid;
                    }
                } else {
                    self.state = GFAState::Finished;
                }
                Some(triple)
            }
        }
    }

    fn get_step(&self, id: PathId, rank: u64) -> Result<NamedNode, ParseError> {
        let path_name = self.graph.get_path_name_vec(id).ok_or(ParseError::Io(Error::new(ErrorKind::Other, "Path should have name")))?;
        let path_name = std::str::from_utf8(&path_name).expect("Should be parsable path name").replace("#", "/");
        let text = format!("{}/path/{}/step/{}", self.base.as_ref().ok_or(ParseError::Io(Error::new(ErrorKind::Other, "Invalid base")))?, path_name, rank);
        let sub = NamedNode::new(text).expect("Named path node should always be fine");
        Ok(sub)
    }

    fn get_path(&self, id: PathId) -> Result<NamedNode, ParseError> {
        let path_name = self.graph.get_path_name_vec(id).ok_or(ParseError::Io(Error::new(ErrorKind::Other, "Path should have name")))?;
        let path_name = std::str::from_utf8(&path_name).expect("Should be parsable path name").replace("#", "/");
        let text = format!("{}/path/{}", self.base.as_ref().ok_or(ParseError::Io(Error::new(ErrorKind::Other, "Invalid base")))?, path_name);
        let sub = NamedNode::new(text).expect("Named path node should always be fine");
        Ok(sub)
    }

    fn get_node(&self, handle: Handle) -> Result<NamedNode, ParseError> {
        let id = handle.unpack_number();
        let text = format!("{}/node/{}", self.base.as_ref().ok_or(ParseError::Io(Error::new(ErrorKind::Other, "Invalid base")))?, id);
        let sub = NamedNode::new(text).expect("Named node should always be fine");
        Ok(sub)
    }

    fn get_step_path_triple(&self, id: PathId, rank: u64) -> Result<Triple, ParseError> {
        let sub = self.get_step(id, rank)?;
        let pre = vg::PATH_PRED;
        let obj = self.get_path(id)?;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_step_node_triple(&self, id: PathId, rank: u64, step: StepPtr) -> Result<Triple, ParseError> {
        let sub = self.get_step(id, rank)?;
        let pre = vg::NODE_PRED;
        let obj = self.get_node(self.graph.path_handle_at_step(id, step).expect("Step has node handle"))?;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_step_rank_triple(&self, id: PathId, rank: u64) -> Result<Triple, ParseError> {
        let sub = self.get_step(id, rank)?;
        let pre = vg::RANK;
        let obj = Literal::new_simple_literal(rank.to_string());
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_step_type_triple(&self, id: PathId, rank: u64) -> Result<Triple, ParseError> {
        let sub = self.get_step(id, rank)?;
        let pre = rdf::TYPE;
        let obj = vg::STEP;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_path_type_triple(&self, id: PathId) -> Result<Triple, ParseError> {
        let sub = self.get_path(id)?;
        let pre = rdf::TYPE;
        let obj = vg::PATH;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_edge_link_directional_triple(&self, edge: Edge) -> Result<Triple, ParseError> {
        let sub = self.get_node(edge.0)?;
        let pre = match (edge.0.is_reverse(), edge.1.is_reverse()) {
            (false, false) => vg::LINKS_FORWARD_TO_FORWARD,
            (false, true) => vg::LINKS_FORWARD_TO_REVERSE,
            (true, false) => vg::LINKS_REVERSE_TO_FORWARD,
            (true, true) => vg::LINKS_REVERSE_TO_REVERSE,
        };
        let obj = self.get_node(edge.1)?;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_edge_link_triple(&self, edge: Edge) -> Result<Triple, ParseError> {
        let sub = self.get_node(edge.0)?;
        let pre = vg::LINKS;
        let obj = self.get_node(edge.1)?;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_node_value_triple(&self, handle: Handle) -> Result<Triple, ParseError> {
        let sub = self.get_node(handle)?;
        let pre = rdf::VALUE;
        let sequence = self.graph.sequence_vec(handle);
        let value = std::str::from_utf8(&sequence).expect("Sequence should be fine, I guess?");
        let obj = Literal::new_simple_literal(value);
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }

    fn get_node_type_triple(&self, handle: Handle) -> Result<Triple, ParseError> {
        let sub = self.get_node(handle)?;
        let pre = rdf::TYPE;
        let obj = vg::NODE;
        let triple = Triple::new(sub, pre, obj);
        Ok(triple)
    }
    // /// Adds some extra bytes to the parser. Should be called when [`read_next`](Self::read_next) returns [`None`] and there is still unread data.
    // pub fn extend_from_slice(&mut self, other: &[u8]) {
    //     self.parser.extend_from_slice(other)
    // }

    // /// Tell the parser that the file is finished.
    // ///
    // /// This triggers the parsing of the final bytes and might lead [`read_next`](Self::read_next) to return some extra values.
    // pub fn end(&mut self) {
    //     self.parser.end()
    // }

    // /// Returns if the parsing is finished i.e. [`end`](Self::end) has been called and [`read_next`](Self::read_next) is always going to return `None`.
    // pub fn is_end(&self) -> bool {
    //     self.parser.is_end()
    // }

    // /// Attempt to parse a new triple from the already provided data.
    // ///
    // /// Returns [`None`] if the parsing is finished or more data is required.
    // /// If it is the case more data should be fed using [`extend_from_slice`](Self::extend_from_slice).
    // pub fn read_next(&mut self) -> Option<Result<Triple, SyntaxError>> {
    //     Some(self.parser.read_next()?.map(Into::into))
    // }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse();
    /// reader.extend_from_slice(file);
    /// assert_eq!(reader.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// reader.read_next().unwrap()?; // We read the first triple
    /// assert_eq!(
    ///     reader.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> GFAPrefixesIter<'_> {
        GFAPrefixesIter {
            inner: self.prefixes.iter(),
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxttl::GFAParser;
    ///
    /// let file = br#"@base <http://example.com/> .
    /// @prefix schema: <http://schema.org/> .
    /// <foo> a schema:Person ;
    ///     schema:name "Foo" ."#;
    ///
    /// let mut reader = GFAParser::new().parse();
    /// reader.extend_from_slice(file);
    /// assert!(reader.base_iri().is_none()); // No base IRI at the beginning
    ///
    /// reader.read_next().unwrap()?; // We read the first triple
    /// assert_eq!(reader.base_iri(), Some("http://example.com/")); // There is now a base IRI
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.base
            .as_ref()
            .map(Iri::as_str)
    }
}

enum GFAState {
    Start,
    NodeType(IntoIter<Handle>),
    NodeValue(IntoIter<Handle>),
    EdgeLink(IntoIter<Edge>),
    EdgeLinkDirectional(IntoIter<Edge>),
    PathType(IntoIter<PathId>),
    StepType(IntoIter<PathId>, PathId, StepPtr, u64),
    StepRank(IntoIter<PathId>, PathId, StepPtr, u64),
    StepNode(IntoIter<PathId>, PathId, StepPtr, u64),
    StepPath(IntoIter<PathId>, PathId, StepPtr, u64),
    Invalid,
    Finished,
}

/// Iterator on the file prefixes.
///
/// See [`LowLevelGFAReader::prefixes`].
pub struct GFAPrefixesIter<'a> {
    inner: Iter<'a, String, Iri<String>>,
}

impl<'a> Iterator for GFAPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (key, value) = self.inner.next()?;
        Some((key.as_str(), value.as_str()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// A [GFA](https://www.w3.org/TR/turtle/) serializer.
///
/// Support for [GFA-star](https://w3c.github.io/rdf-star/cg-spec/2021-12-17.html#turtle-star) is available behind the `rdf-star` feature.
///
/// ```
/// use oxrdf::{NamedNodeRef, TripleRef};
/// use oxttl::GFASerializer;
///
/// let mut writer = GFASerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .serialize_to_write(Vec::new());
/// writer.write_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
///     NamedNodeRef::new("http://schema.org/Person")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     writer.finish()?.as_slice()
/// );
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default)]
#[must_use]
pub struct GFASerializer {
    inner: TriGSerializer,
}

impl GFASerializer {
    /// Builds a new [`GFASerializer`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        self.inner = self.inner.with_prefix(prefix_name, prefix_iri)?;
        Ok(self)
    }

    /// Writes a GFA file to a [`Write`] implementation.
    ///
    /// ```
    /// use oxrdf::{NamedNodeRef, TripleRef};
    /// use oxttl::GFASerializer;
    ///
    /// let mut writer = GFASerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .serialize_to_write(Vec::new());
    /// writer.write_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
    ///     NamedNodeRef::new("http://schema.org/Person")?,
    /// ))?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
    ///     writer.finish()?.as_slice()
    /// );
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn serialize_to_write<W: Write>(self, write: W) -> ToWriteGFAWriter<W> {
        ToWriteGFAWriter {
            inner: self.inner.serialize_to_write(write),
        }
    }

    /// Writes a GFA file to a [`AsyncWrite`] implementation.
    ///
    /// ```
    /// use oxrdf::{NamedNodeRef, TripleRef};
    /// use oxttl::GFASerializer;
    ///
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(),Box<dyn std::error::Error>> {
    /// let mut writer = GFASerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .serialize_to_tokio_async_write(Vec::new());
    /// writer
    ///     .write_triple(TripleRef::new(
    ///         NamedNodeRef::new_unchecked("http://example.com#me"),
    ///         NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
    ///         NamedNodeRef::new_unchecked("http://schema.org/Person"),
    ///     ))
    ///     .await?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
    ///     writer.finish().await?.as_slice()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-tokio")]
    pub fn serialize_to_tokio_async_write<W: AsyncWrite + Unpin>(
        self,
        write: W,
    ) -> ToTokioAsyncWriteGFAWriter<W> {
        ToTokioAsyncWriteGFAWriter {
            inner: self.inner.serialize_to_tokio_async_write(write),
        }
    }

    /// Builds a low-level GFA writer.
    ///
    /// ```
    /// use oxrdf::{NamedNodeRef, TripleRef};
    /// use oxttl::GFASerializer;
    ///
    /// let mut buf = Vec::new();
    /// let mut writer = GFASerializer::new()
    ///     .with_prefix("schema", "http://schema.org/")?
    ///     .serialize();
    /// writer.write_triple(
    ///     TripleRef::new(
    ///         NamedNodeRef::new("http://example.com#me")?,
    ///         NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
    ///         NamedNodeRef::new("http://schema.org/Person")?,
    ///     ),
    ///     &mut buf,
    /// )?;
    /// writer.finish(&mut buf)?;
    /// assert_eq!(
    ///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
    ///     buf.as_slice()
    /// );
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn serialize(self) -> LowLevelGFAWriter {
        LowLevelGFAWriter {
            inner: self.inner.serialize(),
        }
    }
}

/// Writes a GFA file to a [`Write`] implementation. Can be built using [`GFASerializer::serialize_to_write`].
///
/// ```
/// use oxrdf::{NamedNodeRef, TripleRef};
/// use oxttl::GFASerializer;
///
/// let mut writer = GFASerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .serialize_to_write(Vec::new());
/// writer.write_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
///     NamedNodeRef::new("http://schema.org/Person")?,
/// ))?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     writer.finish()?.as_slice()
/// );
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct ToWriteGFAWriter<W: Write> {
    inner: ToWriteTriGWriter<W>,
}

impl<W: Write> ToWriteGFAWriter<W> {
    /// Writes an extra triple.
    pub fn write_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        self.inner
            .write_quad(t.into().in_graph(GraphNameRef::DefaultGraph))
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub fn finish(self) -> io::Result<W> {
        self.inner.finish()
    }
}

/// Writes a GFA file to a [`AsyncWrite`] implementation. Can be built using [`GFASerializer::serialize_to_tokio_async_write`].
///
/// ```
/// use oxrdf::{NamedNodeRef, TripleRef};
/// use oxttl::GFASerializer;
///
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut writer = GFASerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .serialize_to_tokio_async_write(Vec::new());
/// writer
///     .write_triple(TripleRef::new(
///         NamedNodeRef::new_unchecked("http://example.com#me"),
///         NamedNodeRef::new_unchecked("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
///         NamedNodeRef::new_unchecked("http://schema.org/Person"),
///     ))
///     .await?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     writer.finish().await?.as_slice()
/// );
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct ToTokioAsyncWriteGFAWriter<W: AsyncWrite + Unpin> {
    inner: ToTokioAsyncWriteTriGWriter<W>,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> ToTokioAsyncWriteGFAWriter<W> {
    /// Writes an extra triple.
    pub async fn write_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        self.inner
            .write_quad(t.into().in_graph(GraphNameRef::DefaultGraph))
            .await
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub async fn finish(self) -> io::Result<W> {
        self.inner.finish().await
    }
}

/// Writes a GFA file by using a low-level API. Can be built using [`GFASerializer::serialize`].
///
/// ```
/// use oxrdf::{NamedNodeRef, TripleRef};
/// use oxttl::GFASerializer;
///
/// let mut buf = Vec::new();
/// let mut writer = GFASerializer::new()
///     .with_prefix("schema", "http://schema.org/")?
///     .serialize();
/// writer.write_triple(
///     TripleRef::new(
///         NamedNodeRef::new("http://example.com#me")?,
///         NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
///         NamedNodeRef::new("http://schema.org/Person")?,
///     ),
///     &mut buf,
/// )?;
/// writer.finish(&mut buf)?;
/// assert_eq!(
///     b"@prefix schema: <http://schema.org/> .\n<http://example.com#me> a schema:Person .\n",
///     buf.as_slice()
/// );
/// # Result::<_,Box<dyn std::error::Error>>::Ok(())
/// ```
pub struct LowLevelGFAWriter {
    inner: LowLevelTriGWriter,
}

impl LowLevelGFAWriter {
    /// Writes an extra triple.
    pub fn write_triple<'a>(
        &mut self,
        t: impl Into<TripleRef<'a>>,
        write: impl Write,
    ) -> io::Result<()> {
        self.inner
            .write_quad(t.into().in_graph(GraphNameRef::DefaultGraph), write)
    }

    /// Finishes to write the file.
    pub fn finish(&mut self, write: impl Write) -> io::Result<()> {
        self.inner.finish(write)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::panic_in_result_fn)]

    use super::*;
    use oxrdf::{BlankNodeRef, LiteralRef, NamedNodeRef};

    #[test]
    fn test_write() -> io::Result<()> {
        let mut writer = GFASerializer::new().serialize_to_write(Vec::new());
        writer.write_triple(TripleRef::new(
            NamedNodeRef::new_unchecked("http://example.com/s"),
            NamedNodeRef::new_unchecked("http://example.com/p"),
            NamedNodeRef::new_unchecked("http://example.com/o"),
        ))?;
        writer.write_triple(TripleRef::new(
            NamedNodeRef::new_unchecked("http://example.com/s"),
            NamedNodeRef::new_unchecked("http://example.com/p"),
            LiteralRef::new_simple_literal("foo"),
        ))?;
        writer.write_triple(TripleRef::new(
            NamedNodeRef::new_unchecked("http://example.com/s"),
            NamedNodeRef::new_unchecked("http://example.com/p2"),
            LiteralRef::new_language_tagged_literal_unchecked("foo", "en"),
        ))?;
        writer.write_triple(TripleRef::new(
            BlankNodeRef::new_unchecked("b"),
            NamedNodeRef::new_unchecked("http://example.com/p2"),
            BlankNodeRef::new_unchecked("b2"),
        ))?;
        assert_eq!(String::from_utf8(writer.finish()?).unwrap(), "<http://example.com/s> <http://example.com/p> <http://example.com/o> , \"foo\" ;\n\t<http://example.com/p2> \"foo\"@en .\n_:b <http://example.com/p2> _:b2 .\n");
        Ok(())
    }
}
