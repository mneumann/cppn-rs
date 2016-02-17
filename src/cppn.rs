use super::ActivationFunction;
use acyclic_network::{NodeType, Network, Link, Node};
pub use acyclic_network::NodeIndex as CppnNodeIndex;

pub enum CppnNodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

impl NodeType for CppnNodeType {
    fn accept_incoming_links(&self) -> bool {
        match *self {
            CppnNodeType::Input | CppnNodeType::Bias => false,
            _ => true,
        }
    }

    fn accept_outgoing_links(&self) -> bool {
        match *self {
            CppnNodeType::Output => false,
            _ => true,
        }
    }
}

pub type CppnLink = Link<f64>;
pub type CppnNode = Node<CppnNodeType, Box<ActivationFunction>, f64>;
pub type CppnGraph = Network<CppnNodeType, Box<ActivationFunction>, f64>;

/// Represents the output value of a CppnNode.
#[derive(Debug, Copy, Clone)]
enum CppnNodeValue {
    None,
    InCalculation,
    Cached(f64),
}

struct CppnState {
    values: Vec<CppnNodeValue>,
}

impl CppnState {
    fn new(n: usize) -> CppnState {
        CppnState { values: (0..n).map(|_| CppnNodeValue::None).collect() }
    }

    fn set(&mut self, node_idx: CppnNodeIndex, value: f64) {
        self.values[node_idx.index()] = CppnNodeValue::Cached(value);
    }

    fn reset(&mut self) {
        for value in self.values.iter_mut() {
            *value = CppnNodeValue::None;
        }
    }

    /// Calculates the output of node `node_idx`, recursively calculating
    /// the outputs of all dependent nodes.
    /// Panics if it hits a cycle.
    /// XXX: Implement a non-recursive version.
    fn calculate_output_of_node(&mut self, graph: &CppnGraph, node_idx: CppnNodeIndex) -> f64 {
        match self.values[node_idx.index()] {
            CppnNodeValue::InCalculation => {
                panic!("Cycle detected");
            }
            CppnNodeValue::Cached(value) => {
                // Return the cached value.
                return value;
            }
            CppnNodeValue::None => {
                // fall through.
            }
        }
        let node = &graph.nodes()[node_idx.index()];

        // Sum the input links
        let input_sum: f64 = match node.node_type {
            CppnNodeType::Bias => 1.0,
            CppnNodeType::Input => {
                debug_assert!(node.input_links.is_empty());
                panic!("No input set for Input node");
            }
            CppnNodeType::Output | CppnNodeType::Hidden => {
                // Mark this node as being processes. If we hit such a node during a recursive call
                // we have a cycle.
                self.values[node_idx.index()] = CppnNodeValue::InCalculation;
                let mut sum: f64 = 0.0;
                for in_link in &node.input_links {
                    sum += in_link.weight * self.calculate_output_of_node(graph, in_link.node_idx);
                }
                sum
            }
        };

        // apply activation function on `input_sum` (activation function is stored in `node_data`)
        let output = node.node_data.calculate(input_sum);

        // cache the value. this also resets the InCalculation state.
        self.values[node_idx.index()] = CppnNodeValue::Cached(output);

        output
    }
}

/// Represents a Compositional Pattern Producing Network (CPPN)
pub struct Cppn {
    graph: CppnGraph,
    inputs: Vec<CppnNodeIndex>,
    outputs: Vec<CppnNodeIndex>,
    state: CppnState,
}

impl Cppn {
    pub fn new(graph: CppnGraph) -> Cppn {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for (i, node) in graph.nodes().iter().enumerate() {
            match node.node_type {
                CppnNodeType::Input => {
                    inputs.push(CppnNodeIndex::new(i));
                }
                CppnNodeType::Output => {
                    outputs.push(CppnNodeIndex::new(i));
                }
                _ => {}
            }
        }

        let state = CppnState::new(graph.nodes().len());
        Cppn {
            graph: graph,
            inputs: inputs,
            outputs: outputs,
            state: state,
        }
    }

    /// Calculate all outputs
    pub fn calculate(&mut self, inputs: &[&[f64]]) -> Vec<f64> {
        let mut state = &mut self.state;

        state.reset();

        // assign all inputs
        let mut i = 0;
        for input_list in inputs.iter() {
            for &input in input_list.iter() {
                state.set(self.inputs[i], input);
                i += 1;
            }
        }
        assert!(i == self.inputs.len());

        let graph = &self.graph;
        self.outputs
            .iter()
            .map(|&node_idx| state.calculate_output_of_node(graph, node_idx))
            .collect()
    }
}


#[test]
fn test_cycle() {
    use super::Identity;
    let mut g = CppnGraph::new();
    let i1 = g.add_node(CppnNodeType::Input, Box::new(Identity));
    let h1 = g.add_node(CppnNodeType::Hidden, Box::new(Identity));
    let h2 = g.add_node(CppnNodeType::Hidden, Box::new(Identity));
    assert_eq!(true, g.valid_link(i1, i1).is_err());
    assert_eq!(true, g.valid_link(h1, h1).is_err());

    assert_eq!(true, g.valid_link(h1, i1).is_err());
    assert_eq!(Ok(()), g.valid_link(i1, h1));
    assert_eq!(Ok(()), g.valid_link(i1, h2));
    assert_eq!(Ok(()), g.valid_link(h1, h2));

    g.add_link(i1, h1, 0.0);
    assert_eq!(true, g.link_would_cycle(h1, i1));
    assert_eq!(false, g.link_would_cycle(i1, h1));
    assert_eq!(false, g.link_would_cycle(i1, h2));
    assert_eq!(true, g.link_would_cycle(i1, i1));
    assert_eq!(false, g.link_would_cycle(h1, h2));
    assert_eq!(false, g.link_would_cycle(h2, h1));
    assert_eq!(false, g.link_would_cycle(h2, i1));

    g.add_link(h1, h2, 0.0);
    assert_eq!(true, g.link_would_cycle(h2, i1));
    assert_eq!(true, g.link_would_cycle(h1, i1));
    assert_eq!(true, g.link_would_cycle(h2, h1));
    assert_eq!(false, g.link_would_cycle(i1, h2));
}

#[test]
fn test_simple_cppn() {
    use super::Identity;
    use super::bipolar::Linear;

    let mut g = CppnGraph::new();
    let i1 = g.add_node(CppnNodeType::Input, Box::new(Identity));
    let h1 = g.add_node(CppnNodeType::Hidden, Box::new(Linear));
    let o1 = g.add_node(CppnNodeType::Output, Box::new(Identity));
    g.add_link(i1, h1, 0.5);
    g.add_link(h1, o1, 1.0);

    let mut cppn = Cppn::new(g);

    assert_eq!(vec![0.5 * 0.5], cppn.calculate(&[&[0.5]]));
    assert_eq!(vec![1.0], cppn.calculate(&[&[4.0]]));
    assert_eq!(vec![-1.0], cppn.calculate(&[&[-4.0]]));
}


#[test]
fn test_find_random_unconnected_link_no_cycle() {
    use rand;
    use super::Identity;

    let mut g = CppnGraph::new();
    let i1 = g.add_node(CppnNodeType::Input, Box::new(Identity));
    let o1 = g.add_node(CppnNodeType::Output, Box::new(Identity));
    let o2 = g.add_node(CppnNodeType::Output, Box::new(Identity));

    let mut rng = rand::thread_rng();

    let link = g.find_random_unconnected_link_no_cycle(&mut rng);
    assert_eq!(true, link.is_some());
    let l = link.unwrap();
    assert!((i1, o1) == l || (i1, o2) == l);

    g.add_link(i1, o2, 0.0);
    let link = g.find_random_unconnected_link_no_cycle(&mut rng);
    assert_eq!(true, link.is_some());
    assert_eq!((i1, o1), link.unwrap());

    g.add_link(i1, o1, 0.0);
    let link = g.find_random_unconnected_link_no_cycle(&mut rng);
    assert_eq!(false, link.is_some());
}
