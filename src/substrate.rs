use position::Position;
use cppn::{Cppn, CppnNodeType};

/// Represents a node in the substrate. `T` is an arbitrary
/// type used to store additional information about that node.
pub struct Node<P: Position, T> {
    pub position: P,
    pub data: T,
}

pub struct Substrate<P: Position, T> {
    nodes: Vec<Node<P, T>>,
}

pub struct LinkIterator<'a, N: CppnNodeType + 'a, P: Position + 'a, T: 'a> {
    nodes: &'a [Node<P, T>],
    cppn: &'a mut Cppn<'a, N>,
    inner: usize,
    outer: usize,
    max_distance: Option<f64>,
}

#[derive(Copy, Clone)]
pub struct Link<'a, P: Position + 'a, T: 'a> {
    pub source: &'a Node<P, T>,
    pub target: &'a Node<P, T>,
    pub source_idx: usize,
    pub target_idx: usize,
    pub weight: f64,
    pub distance: f64,
}

impl<'a, N: CppnNodeType + 'a, P: Position + 'a, T: 'a> Iterator for LinkIterator<'a, N, P, T> {
    type Item = Link<'a, P, T>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.inner >= self.nodes.len() {
                self.inner = 0;
                self.outer += 1;
            }
            if self.outer >= self.nodes.len() {
                return None;
            }
            if self.inner == self.outer {
                self.inner += 1;
                continue;
            }

            assert!(self.inner < self.nodes.len() && self.outer < self.nodes.len() &&
                    self.inner != self.outer);
            let source = &self.nodes[self.inner];
            let target = &self.nodes[self.outer];
            let distance = source.position.distance(&target.position);

            // reject a pair of nodes based on `max_distance`.
            if let Some(max_d) = self.max_distance {
                if distance > max_d {
                    self.inner += 1;
                    continue;
                }
            }

            // Calculate the weight between source and target using the CPPN.
            let inputs_to_cppn = [source.position.coords(), target.position.coords()];
            let outputs_from_cppn = self.cppn.calculate(&inputs_to_cppn);
            assert!(outputs_from_cppn.len() == 1);
            let weight = outputs_from_cppn[0];

            let link = Link {
                source: source,
                target: target,
                source_idx: self.inner,
                target_idx: self.outer,
                weight: weight,
                distance: distance,
            };
            self.inner += 1;
            return Some(link);
        }
    }
}

impl<P: Position, T> Substrate<P, T> {
    pub fn new() -> Substrate<P, T> {
        Substrate { nodes: Vec::new() }
    }

    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }

    pub fn add_node(&mut self, position: P, data: T) {
        self.nodes.push(Node {
            position: position,
            data: data,
        });
    }

    /// Iterate over all produced links of Cppn.
    pub fn iter_links<'a, N>(&'a self,
                             cppn: &'a mut Cppn<'a, N>,
                             max_distance: Option<f64>)
                             -> LinkIterator<'a, N, P, T>
        where N: CppnNodeType
    {
        LinkIterator {
            nodes: &self.nodes,
            cppn: cppn,
            inner: 0,
            outer: 0,
            max_distance: max_distance,
        }
    }
}
