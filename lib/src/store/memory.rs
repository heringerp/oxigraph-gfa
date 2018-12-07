use crate::store::encoded::*;
use crate::store::numeric_encoder::*;
use crate::utils::MutexPoisonError;
use crate::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;
use url::Url;

/// Memory based implementation of the `rudf::model::Dataset` trait.
/// They are cheap to build using the `MemoryDataset::default()` method.
///
/// Usage example:
/// ```
/// use rudf::model::*;
/// use rudf::store::MemoryDataset;
/// use std::str::FromStr;
///
/// let dataset = MemoryDataset::default();
/// let default_graph = dataset.default_graph();
/// let ex = NamedNode::from_str("http://example.com").unwrap();
/// let triple = Triple::new(ex.clone(), ex.clone(), ex.clone());
/// default_graph.insert(&triple);
/// let results: Vec<Quad> = dataset.quads_for_subject(&ex.into()).unwrap().map(|t| t.unwrap()).collect();
/// assert_eq!(vec![triple.in_graph(None)], results);
/// ```
pub type MemoryDataset = StoreDataset<MemoryStore>;

/// Memory based implementation of the `rudf::model::Graph` trait.
/// They are cheap to build using the `MemoryGraph::default()` method.
///
/// Usage example:
/// ```
/// use rudf::model::*;
/// use rudf::store::MemoryGraph;
/// use std::str::FromStr;
///
/// let graph = MemoryGraph::default();
/// let ex = NamedNode::from_str("http://example.com").unwrap();
/// let triple = Triple::new(ex.clone(), ex.clone(), ex.clone());
/// graph.insert(&triple);
/// let results: Vec<Triple> = graph.triples_for_subject(&ex.into()).unwrap().map(|t| t.unwrap()).collect();
/// assert_eq!(vec![triple], results);
/// ```
pub type MemoryGraph = StoreDefaultGraph<MemoryStore>;

pub struct MemoryStore {
    string_store: MemoryStringStore,
    graph_indexes: RwLock<BTreeMap<EncodedTerm, MemoryGraphIndexes>>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self {
            string_store: MemoryStringStore::default(),
            graph_indexes: RwLock::default(),
        }
    }
}

#[derive(Default)]
struct MemoryGraphIndexes {
    spo: BTreeMap<EncodedTerm, BTreeMap<EncodedTerm, BTreeSet<EncodedTerm>>>,
    pos: BTreeMap<EncodedTerm, BTreeMap<EncodedTerm, BTreeSet<EncodedTerm>>>,
    osp: BTreeMap<EncodedTerm, BTreeMap<EncodedTerm, BTreeSet<EncodedTerm>>>,
}

impl StringStore for MemoryStore {
    fn insert_str(&self, value: &str) -> Result<u64> {
        self.string_store.insert_str(value)
    }

    fn get_str(&self, id: u64) -> Result<String> {
        self.string_store.get_str(id)
    }

    fn get_url(&self, id: u64) -> Result<Url> {
        self.string_store.get_url(id)
    }
}

impl EncodedQuadsStore for MemoryStore {
    type QuadsIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectPredicateIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectPredicateObjectIterator =
        <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectObjectIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForPredicateIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForPredicateObjectIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForObjectIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForGraphIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectGraphIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectPredicateGraphIterator =
        <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForSubjectObjectGraphIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForPredicateGraphIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForPredicateObjectGraphIterator =
        <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;
    type QuadsForObjectGraphIterator = <Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter;

    fn quads(&self) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            for (s, pos) in &graph.spo {
                for (p, os) in pos.iter() {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(*s, *p, *o, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject(
        &self,
        subject: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(pos) = graph.spo.get(&subject) {
                for (p, os) in pos.iter() {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(subject, *p, *o, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_predicate(
        &self,
        subject: EncodedTerm,
        predicate: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(pos) = graph.spo.get(&subject) {
                if let Some(os) = pos.get(&predicate) {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(subject, predicate, *o, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_predicate_object(
        &self,
        subject: EncodedTerm,
        predicate: EncodedTerm,
        object: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(pos) = graph.spo.get(&subject) {
                if let Some(os) = pos.get(&predicate) {
                    if os.contains(&object) {
                        result.push(Ok(EncodedQuad::new(
                            subject,
                            predicate,
                            object,
                            *graph_name,
                        )))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_object(
        &self,
        subject: EncodedTerm,
        object: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(sps) = graph.osp.get(&object) {
                if let Some(ps) = sps.get(&subject) {
                    for p in ps.iter() {
                        result.push(Ok(EncodedQuad::new(subject, *p, object, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_predicate(
        &self,
        predicate: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(oss) = graph.pos.get(&predicate) {
                for (o, ss) in oss.iter() {
                    for s in ss.iter() {
                        result.push(Ok(EncodedQuad::new(*s, predicate, *o, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_predicate_object(
        &self,
        predicate: EncodedTerm,
        object: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(oss) = graph.pos.get(&predicate) {
                if let Some(ss) = oss.get(&object) {
                    for s in ss.iter() {
                        result.push(Ok(EncodedQuad::new(*s, predicate, object, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_object(
        &self,
        object: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        for (graph_name, graph) in self.graph_indexes()?.iter() {
            if let Some(sps) = graph.osp.get(&object) {
                for (s, ps) in sps.iter() {
                    for p in ps.iter() {
                        result.push(Ok(EncodedQuad::new(*s, *p, object, *graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_graph(
        &self,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            for (s, pos) in &graph.spo {
                for (p, os) in pos.iter() {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(*s, *p, *o, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_graph(
        &self,
        subject: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(pos) = graph.spo.get(&subject) {
                for (p, os) in pos.iter() {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(subject, *p, *o, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_predicate_graph(
        &self,
        subject: EncodedTerm,
        predicate: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(pos) = graph.spo.get(&subject) {
                if let Some(os) = pos.get(&predicate) {
                    for o in os.iter() {
                        result.push(Ok(EncodedQuad::new(subject, predicate, *o, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_subject_object_graph(
        &self,
        subject: EncodedTerm,
        object: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(sps) = graph.osp.get(&object) {
                if let Some(ps) = sps.get(&subject) {
                    for p in ps.iter() {
                        result.push(Ok(EncodedQuad::new(subject, *p, object, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_predicate_graph(
        &self,
        predicate: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(oss) = graph.pos.get(&predicate) {
                for (o, ss) in oss.iter() {
                    for s in ss.iter() {
                        result.push(Ok(EncodedQuad::new(*s, predicate, *o, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_predicate_object_graph(
        &self,
        predicate: EncodedTerm,
        object: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(oss) = graph.pos.get(&predicate) {
                if let Some(ss) = oss.get(&object) {
                    for s in ss.iter() {
                        result.push(Ok(EncodedQuad::new(*s, predicate, object, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn quads_for_object_graph(
        &self,
        object: EncodedTerm,
        graph_name: EncodedTerm,
    ) -> Result<<Vec<Result<EncodedQuad>> as IntoIterator>::IntoIter> {
        let mut result = Vec::default();
        if let Some(graph) = self.graph_indexes()?.get(&graph_name) {
            if let Some(sps) = graph.osp.get(&object) {
                for (s, ps) in sps.iter() {
                    for p in ps.iter() {
                        result.push(Ok(EncodedQuad::new(*s, *p, object, graph_name)))
                    }
                }
            }
        }
        Ok(result.into_iter())
    }

    fn contains(&self, quad: &EncodedQuad) -> Result<bool> {
        Ok(self
            .graph_indexes()?
            .get(&quad.graph_name)
            .map_or(false, |graph| {
                graph.spo.get(&quad.subject).map_or(false, |po| {
                    po.get(&quad.predicate)
                        .map_or(false, |o| o.contains(&quad.object))
                })
            }))
    }

    fn insert(&self, quad: &EncodedQuad) -> Result<()> {
        let mut graph_indexes = self.graph_indexes_mut()?;
        let graph = graph_indexes
            .entry(quad.graph_name)
            .or_insert_with(MemoryGraphIndexes::default);
        graph
            .spo
            .entry(quad.subject)
            .or_default()
            .entry(quad.predicate)
            .or_default()
            .insert(quad.object);
        graph
            .pos
            .entry(quad.predicate)
            .or_default()
            .entry(quad.object)
            .or_default()
            .insert(quad.subject);
        graph
            .osp
            .entry(quad.object)
            .or_default()
            .entry(quad.subject)
            .or_default()
            .insert(quad.predicate);
        Ok(())
    }

    fn remove(&self, quad: &EncodedQuad) -> Result<()> {
        let mut graph_indexes = self.graph_indexes_mut()?;
        let mut empty_graph = false;
        if let Some(graph) = graph_indexes.get_mut(&quad.graph_name) {
            {
                let mut empty_pos = false;
                if let Some(pos) = graph.spo.get_mut(&quad.subject) {
                    let mut empty_os = false;
                    if let Some(os) = pos.get_mut(&quad.predicate) {
                        os.remove(&quad.object);
                        empty_os = os.is_empty();
                    }
                    if empty_os {
                        pos.remove(&quad.predicate);
                    }
                    empty_pos = pos.is_empty();
                }
                if empty_pos {
                    graph.spo.remove(&quad.subject);
                }
            }

            {
                let mut empty_oss = false;
                if let Some(oss) = graph.pos.get_mut(&quad.predicate) {
                    let mut empty_ss = false;
                    if let Some(ss) = oss.get_mut(&quad.object) {
                        ss.remove(&quad.subject);
                        empty_ss = ss.is_empty();
                    }
                    if empty_ss {
                        oss.remove(&quad.object);
                    }
                    empty_oss = oss.is_empty();
                }
                if empty_oss {
                    graph.pos.remove(&quad.predicate);
                }
            }

            {
                let mut empty_sps = false;
                if let Some(sps) = graph.osp.get_mut(&quad.object) {
                    let mut empty_ps = false;
                    if let Some(ps) = sps.get_mut(&quad.subject) {
                        ps.remove(&quad.predicate);
                        empty_ps = ps.is_empty();
                    }
                    if empty_ps {
                        sps.remove(&quad.subject);
                    }
                    empty_sps = sps.is_empty();
                }
                if empty_sps {
                    graph.osp.remove(&quad.object);
                }
            }

            empty_graph = graph.spo.is_empty();
        }
        if empty_graph {
            graph_indexes.remove(&quad.graph_name);
        }
        Ok(())
    }
}

impl MemoryStore {
    fn graph_indexes(
        &self,
    ) -> Result<RwLockReadGuard<'_, BTreeMap<EncodedTerm, MemoryGraphIndexes>>> {
        Ok(self.graph_indexes.read().map_err(MutexPoisonError::from)?)
    }

    fn graph_indexes_mut(
        &self,
    ) -> Result<RwLockWriteGuard<'_, BTreeMap<EncodedTerm, MemoryGraphIndexes>>> {
        Ok(self.graph_indexes.write().map_err(MutexPoisonError::from)?)
    }
}
