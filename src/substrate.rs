use position::Position;
use cppn::{Cppn, CppnNodeType};
use acyclic_network::NodeType;
use std::fmt::Debug;

/// Represents a node in the substrate. `T` stores additional information about that node.

pub struct Node<P, T>
    where P: Position,
          T: NodeType
{
    pub position: P,
    pub node_type: T,
}

pub struct Substrate<P, T>
    where P: Position,
          T: NodeType
{
    nodes: Vec<Node<P, T>>,
}

#[derive(Clone)]
pub struct Link<'a, P, T>
    where P: Position + 'a,
          T: NodeType + 'a
{
    pub source: &'a Node<P, T>,
    pub target: &'a Node<P, T>,
    pub source_idx: usize,
    pub target_idx: usize,
    pub outputs: Vec<f64>,
    pub distance: f64,
}

pub enum LinkMode {
    AbsolutePositions,
    AbsolutePositionsAndDistance,
    RelativePositionOfTarget,
}

impl<P, T> Substrate<P, T>
    where P: Position,
          T: NodeType
{
    pub fn new() -> Self {
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

    pub fn each_link<'a, N, L, EXTID, F>(&'a self,
                                         cppn: &'a mut Cppn<'a, N, L, EXTID>,
                                         max_distance: Option<f64>,
                                         mode: LinkMode,
                                         callback: &mut F)
        where N: CppnNodeType,
              L: Copy + Debug + Send + Sized + Into<f64> + 'a,
              EXTID: Copy + Debug + Send + Sized + Ord + 'a,
              F: FnMut(Link<'a, P, T>)
    {
        for (source_idx, source) in self.nodes.iter().enumerate() {
            // Reject invalid connections.
            if !source.node_type.accept_outgoing_links() {
                continue;
            }

            for (target_idx, target) in self.nodes.iter().enumerate() {
                // Reject invalid connections.
                if !target.node_type.accept_incoming_links() {
                    continue;
                }

                let distance = source.position.distance(&target.position);

                // reject a pair of nodes based on `max_distance`.
                if let Some(max_d) = max_distance {
                    if distance > max_d {
                        continue;
                    }
                }

                // Calculate the weight between source and target using the CPPN.

                let outputs_from_cppn = match mode {
                    LinkMode::AbsolutePositions => {
                        let inputs_to_cppn = [source.position.coords(), target.position.coords()];
                        cppn.calculate(&inputs_to_cppn)
                    }
                    LinkMode::AbsolutePositionsAndDistance => {
                        let distance_between: &[_] = &[distance];
                        let inputs_to_cppn = [source.position.coords(),
                                              target.position.coords(),
                                              distance_between];
                        cppn.calculate(&inputs_to_cppn)
                    }
                    LinkMode::RelativePositionOfTarget => {
                        let relative_position_of_target =
                            source.position.relative_position(&target.position);
                        let inputs_to_cppn = [source.position.coords(),
                                              relative_position_of_target.coords()];
                        cppn.calculate(&inputs_to_cppn)
                    }
                };

                let link = Link {
                    source: source,
                    target: target,
                    source_idx: source_idx,
                    target_idx: target_idx,
                    outputs: outputs_from_cppn,
                    distance: distance,
                };
                callback(link);
            }
        }
    }
}
