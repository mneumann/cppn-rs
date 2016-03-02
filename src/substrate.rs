use position::Position;
use cppn::{Cppn, CppnNodeType};
use acyclic_network::NodeType;
use std::fmt::Debug;

/// Represents a node in the substrate. `T` stores  additional information about that node.

pub struct Node<P: Position, T: NodeType> {
    pub position: P,
    pub node_type: T,
}

pub struct Substrate<P: Position, T: NodeType> {
    nodes: Vec<Node<P, T>>,
}

pub struct LinkIterator<'a,
                        N: CppnNodeType + 'a,
                        L: Copy + Debug + Send + Sized + Into<f64> + 'a,
                        EXTID: Copy + Debug + Send + Sized + Ord + 'a,
                        P: Position + 'a,
                        T: NodeType + 'a>
{
    nodes: &'a [Node<P, T>],
    cppn: &'a mut Cppn<'a, N, L, EXTID>,
    inner: usize,
    outer: usize,
    max_distance: Option<f64>,
}

#[derive(Copy, Clone)]
pub struct Link<'a, P: Position + 'a, T: NodeType + 'a> {
    pub source: &'a Node<P, T>,
    pub target: &'a Node<P, T>,
    pub source_idx: usize,
    pub target_idx: usize,
    pub weight: f64,
    pub distance: f64,
}

impl<'a, N: CppnNodeType + 'a,
   L: Copy + Debug + Send + Sized + Into<f64> + 'a,
    EXTID: Copy + Debug + Send + Sized + Ord + 'a,
P: Position + 'a, T: NodeType + 'a> Iterator for LinkIterator<'a, N, L, EXTID, P, T> {
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

// Reject invalid connections.

            if !source.node_type.accept_outgoing_links() || !target.node_type.accept_incoming_links() {
                    self.inner += 1;
                    continue;
            }

            let distance = source.position.distance(&target.position);

// reject a pair of nodes based on `max_distance`.

            if let Some(max_d) = self.max_distance {
                if distance > max_d {
                    self.inner += 1;
                    continue;
                }
            }

// Calculate the weight between source and target using the CPPN.

            let distance_between: &[_] = &[distance];
            let inputs_to_cppn = [source.position.coords(), target.position.coords(), distance_between];
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

impl<P: Position, T: NodeType> Substrate<P, T> {
    pub fn new() -> Substrate<P, T> {
        Substrate { nodes: Vec::new() }
    }

    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }

    pub fn add_node(&mut self, position: P, node_type: T) {
        self.nodes.push(Node {
            position: position,
            node_type: node_type,
        });
    }

    /// Iterate over all produced links of Cppn.
    pub fn iter_links<'a, N, L, EXTID>(&'a self,
                                       cppn: &'a mut Cppn<'a, N, L, EXTID>,
                                       max_distance: Option<f64>)
                                       -> LinkIterator<'a, N, L, EXTID, P, T>
        where N: CppnNodeType,
              L: Copy + Debug + Send + Sized + Into<f64> + 'a,
              EXTID: Copy + Debug + Send + Sized + Ord + 'a
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
