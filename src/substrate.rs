use position::Position;
use cppn::{Cppn, CppnNodeType};
use acyclic_network::NodeType;
use std::fmt::Debug;

/// Represents a node in the substrate. `T` stores additional information about that node.
#[derive(Clone, Debug)]
pub struct Node<P, T>
    where P: Position,
          T: NodeType
{
    pub position: P,
    pub node_type: T,
}

#[derive(Clone, Debug)]
pub struct Layer<P, T>
    where P: Position,
          T: NodeType
{
    nodes: Vec<Node<P, T>>,
}

impl<P, T> Layer<P, T>
    where P: Position,
          T: NodeType
{
    pub fn new() -> Self {
        Layer { nodes: Vec::new() }
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
}

#[derive(Clone)]
pub struct Link<'a, P, T>
    where P: Position + 'a,
          T: NodeType + 'a
{
    pub source: &'a Node<P, T>,
    pub target: &'a Node<P, T>,
    pub source_idx: (usize, usize), // (layer, node)
    pub target_idx: (usize, usize), // (layer, node)
    pub outputs: Vec<f64>,
    pub distance: f64,
}

#[derive(Clone, Debug)]
pub enum LinkMode {
    AbsolutePositions,
    AbsolutePositionsAndDistance,
    RelativePositionOfTarget,
}

#[derive(Clone, Debug)]
struct LayerLink {
    from_layer: usize,
    to_layer: usize,
    max_distance: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Substrate<P, T>
    where P: Position,
          T: NodeType
{
    layers: Vec<Layer<P, T>>,
    layer_links: Vec<LayerLink>,
}


impl<P, T> Substrate<P, T>
    where P: Position,
          T: NodeType
{
    pub fn new() -> Self {
        Substrate {
            layers: Vec::new(),
            layer_links: Vec::new(),
        }
    }

    pub fn layers(&self) -> &[Layer<P, T>] {
        &self.layers
    }

    pub fn add_layer(&mut self, layer: Layer<P, T>) -> usize {
        let layer_idx = self.layers.len();
        self.layers.push(layer);
        return layer_idx;
    }

    pub fn add_layer_link(&mut self,
                          from_layer: usize,
                          to_layer: usize,
                          max_distance: Option<f64>) {
        self.layer_links.push(LayerLink {
            from_layer: from_layer,
            to_layer: to_layer,
            max_distance: max_distance,
        });
    }

    /// Iterate over all produced links of Cppn.

    pub fn each_link<'a, N, L, EXTID, F>(&'a self,
                                         cppn: &'a mut Cppn<'a, N, L, EXTID>,
                                         mode: LinkMode,
                                         callback: &mut F)
        where N: CppnNodeType,
              L: Copy + Debug + Send + Sized + Into<f64> + 'a,
              EXTID: Copy + Debug + Send + Sized + Ord + 'a,
              F: FnMut(Link<'a, P, T>)
    {
        for layer_link in self.layer_links.iter() {
            for (source_idx, source) in self.layers[layer_link.from_layer]
                                            .nodes
                                            .iter()
                                            .enumerate() {
                // Reject invalid connections.
                if !source.node_type.accept_outgoing_links() {
                    continue;
                }

                for (target_idx, target) in self.layers[layer_link.to_layer]
                                                .nodes
                                                .iter()
                                                .enumerate() {
                    // Reject invalid connections.
                    if !target.node_type.accept_incoming_links() {
                        continue;
                    }

                    let distance = source.position.distance(&target.position);

                    // reject a pair of nodes based on `max_distance`.
                    if let Some(max_d) = layer_link.max_distance {
                        if distance > max_d {
                            continue;
                        }
                    }

                    // Calculate the weight between source and target using the CPPN.

                    let outputs_from_cppn = match mode {
                        LinkMode::AbsolutePositions => {
                            let inputs_to_cppn = [source.position.coords(),
                                                  target.position.coords()];
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
                        source_idx: (layer_link.from_layer, source_idx),
                        target_idx: (layer_link.to_layer, target_idx),
                        outputs: outputs_from_cppn,
                        distance: distance,
                    };
                    callback(link);
                }
            }
        }
    }
}
