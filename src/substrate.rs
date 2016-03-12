use position::Position;
use cppn::{Cppn, CppnNodeType};
use std::fmt::Debug;

#[derive(Debug, Copy, Clone)]
pub enum NodeConnectivity {
    In,
    Out,
    InOut
}

/// Represents a node in the substrate. `T` stores additional information about that node.
#[derive(Clone, Debug)]
pub struct Node<P, T>
    where P: Position,
{
    pub position: P,
    pub node_info: T,
    pub node_connectivity: NodeConnectivity
}

#[derive(Clone, Debug)]
pub struct Layer<P, T>
    where P: Position,
{
    nodes: Vec<Node<P, T>>,
}

impl<P, T> Layer<P, T>
    where P: Position,
{
    pub fn new() -> Self {
        Layer { nodes: Vec::new() }
    }

    pub fn nodes(&self) -> &[Node<P, T>] {
        &self.nodes
    }

    pub fn add_node(&mut self, position: P, node_info: T, node_connectivity: NodeConnectivity) {
        self.nodes.push(Node {
            position: position,
            node_info: node_info,
            node_connectivity: node_connectivity,
        });
    }
}

#[derive(Clone)]
pub struct Link<'a, P, T>
    where P: Position + 'a,
          T: 'a
{
    pub source: &'a Node<P, T>,
    pub target: &'a Node<P, T>,
    pub source_idx: (usize, usize), // (layer, node)
    pub target_idx: (usize, usize), // (layer, node)
    pub outputs: Vec<f64>,
    pub distance: f64,
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
{
    layers: Vec<Layer<P, T>>,
    layer_links: Vec<LayerLink>,
}


impl<P, T> Substrate<P, T>
    where P: Position,
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
                match source.node_connectivity {
                    NodeConnectivity::Out | NodeConnectivity::InOut => {} 
                    NodeConnectivity::In => {
                        // Node does not allow outgoing connections
                        continue;
                    }
                }

                for (target_idx, target) in self.layers[layer_link.to_layer]
                                                .nodes
                                                .iter()
                                                .enumerate() {
                    // Reject invalid connections.
                    match target.node_connectivity {
                        NodeConnectivity::In | NodeConnectivity::InOut => {} 
                        NodeConnectivity::Out => {
                            // Node does not allow incoming connections
                            continue;
                        }
                    }

                    let distance = source.position.distance(&target.position);

                    // reject a pair of nodes based on `max_distance`.
                    if let Some(max_d) = layer_link.max_distance {
                        if distance > max_d {
                            continue;
                        }
                    }

                    // Calculate the weight between source and target using the CPPN.

                    let inputs_to_cppn = [source.position.coords(), target.position.coords()];

                    let outputs_from_cppn = cppn.calculate(&inputs_to_cppn);

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
