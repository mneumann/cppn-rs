use super::ActivationFunction;

pub enum CppnNodeType {
    Bias,
    Input,
    Output,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CppnNodeIndex(usize);

impl CppnNodeIndex {
    fn index(&self) -> usize {
        self.0
    }
}

struct CppnLink {
    source_node_idx: CppnNodeIndex,
    weight: f64,
}

struct CppnNode {
    node_type: CppnNodeType,
    activation_function: Box<ActivationFunction>,
    input_links: Vec<CppnLink>,
}

struct CppnGraph {
    nodes: Vec<CppnNode>,
}

impl CppnGraph {
    fn new() -> CppnGraph {
        CppnGraph { nodes: Vec::new() }
    }

    fn add_node(&mut self,
                node_type: CppnNodeType,
                activation_function: Box<ActivationFunction>)
                -> CppnNodeIndex {
        let idx = CppnNodeIndex(self.nodes.len());
        self.nodes.push(CppnNode {
            node_type: node_type,
            activation_function: activation_function,
            input_links: Vec::new(),
        });
        return idx;
    }

    fn add_link(&mut self,
                source_node_idx: CppnNodeIndex,
                target_node_idx: CppnNodeIndex,
                weight: f64) {
        if source_node_idx == target_node_idx {
            panic!("Loops are not allowed");
        }

        if let CppnNodeType::Output = self.nodes[source_node_idx.index()].node_type {
            panic!("Cannot have an outgoing connection from Output node");
        }

        let mut target_node = &mut self.nodes[target_node_idx.index()];

        if let CppnNodeType::Input = target_node.node_type {
            panic!("Cannot have an incoming connection to Input node");
        }

        if let CppnNodeType::Bias = target_node.node_type {
            panic!("Cannot have an incoming connection to Bias node");
        }

        target_node.input_links.push(CppnLink {
            source_node_idx: source_node_idx,
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
    fn reset(&mut self) {
        for value in self.values.iter_mut() {
            *value = CppnNodeValue::None;
        }
    }

    /// Calculates the output of node `node_idx`, recursively calculating
    /// the outputs of all dependent nodes.
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
                    sum += in_link.weight *
                           self.calculate_output_of_node(graph, in_link.source_node_idx);
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
pub struct Cppn;

impl Cppn {}
