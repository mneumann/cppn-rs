use super::ActivationFunction;
use fixedbitset::FixedBitSet;

pub enum CppnNodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CppnNodeIndex(usize);

impl CppnNodeIndex {
    fn index(&self) -> usize {
        self.0
    }
}

struct CppnLink {
    node_idx: CppnNodeIndex,
    weight: f64,
}

struct CppnNode {
    node_type: CppnNodeType,
    activation_function: Box<ActivationFunction>,
    input_links: Vec<CppnLink>,
    output_links: Vec<CppnLink>,
}

pub struct CppnGraph {
    nodes: Vec<CppnNode>,
}

impl CppnGraph {
    pub fn new() -> CppnGraph {
        CppnGraph { nodes: Vec::new() }
    }

    pub fn add_node(&mut self,
                    node_type: CppnNodeType,
                    activation_function: Box<ActivationFunction>)
                    -> CppnNodeIndex {
        let idx = CppnNodeIndex(self.nodes.len());
        self.nodes.push(CppnNode {
            node_type: node_type,
            activation_function: activation_function,
            input_links: Vec::new(),
            output_links: Vec::new(),
        });
        return idx;
    }

    // Returns true if the introduction of this directed link would lead towards a cycle.
    pub fn link_would_cycle(&self,
                            source_node_idx: CppnNodeIndex,
                            target_node_idx: CppnNodeIndex)
                            -> bool {
        if source_node_idx == target_node_idx {
            return true;
        }

        let mut seen_nodes = FixedBitSet::with_capacity(self.nodes.len());
        let mut nodes_to_visit = Vec::new();

        // We start at the target node and iterate all paths from there. If we hit the source node,
        // we found a cycle. Otherwise not.
        let start = target_node_idx.index();

        // We are looking for a path to this node.
        let path_to = source_node_idx.index();

        nodes_to_visit.push(start);
        seen_nodes.insert(start);

        while let Some(visit_node) = nodes_to_visit.pop() {
            for out_link in &self.nodes[visit_node].output_links {
                let next_node = out_link.node_idx.index();
                if !seen_nodes.contains(next_node) {
                    if next_node == path_to {
                        // We found a path to `path_to`. We have found a cycle.
                        return true;
                    }

                    seen_nodes.insert(next_node);
                    nodes_to_visit.push(next_node)
                }
            }
        }

        // We haven't found a cycle.
        return false;
    }

    // Check if the link is valid. Doesn't check for cycles.
    pub fn valid_link(&self,
                      source_node_idx: CppnNodeIndex,
                      target_node_idx: CppnNodeIndex)
                      -> Result<(), &'static str> {
        if source_node_idx == target_node_idx {
            return Err("Loops are not allowed");
        }

        match self.nodes[source_node_idx.index()].node_type {
            CppnNodeType::Output => {
                return Err("Cannot have an outgoing connection from Output node")
            }
            _ => {}
        }

        match self.nodes[target_node_idx.index()].node_type {
            CppnNodeType::Input | CppnNodeType::Bias => {
                return Err("Cannot have an incoming connection to Input/Bias node");
            }
            _ => {}
        }

        Ok(())
    }

    // Note: Doesn't check for cycles (except in the simple reflexive case).
    pub fn add_link(&mut self,
                    source_node_idx: CppnNodeIndex,
                    target_node_idx: CppnNodeIndex,
                    weight: f64) {
        if let Err(err) = self.valid_link(source_node_idx, target_node_idx) {
            panic!(err);
        }

        self.nodes[source_node_idx.index()].output_links.push(CppnLink {
            node_idx: target_node_idx,
            weight: weight,
        });

        self.nodes[target_node_idx.index()].input_links.push(CppnLink {
            node_idx: source_node_idx,
            weight: weight,
        });
    }
}

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
        let node = &graph.nodes[node_idx.index()];

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

        // apply activation function on `input_sum`.
        let output = node.activation_function.calculate(input_sum);

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

        for (i, node) in graph.nodes.iter().enumerate() {
            match node.node_type {
                CppnNodeType::Input => {
                    inputs.push(CppnNodeIndex(i));
                }
                CppnNodeType::Output => {
                    outputs.push(CppnNodeIndex(i));
                }
                _ => {}
            }
        }

        let state = CppnState::new(graph.nodes.len());
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
