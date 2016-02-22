use activation_function::ActivationFunction;
use acyclic_network::{NodeType, Network};
pub use acyclic_network::NodeIndex as CppnNodeIndex;
use fixedbitset::FixedBitSet;

pub trait CppnNodeType: NodeType + ActivationFunction {
    fn is_input_node(&self) -> bool;
    fn is_output_node(&self) -> bool;
}

/// A concrete implementation of a CppnNodeType.
#[derive(Clone, Debug)]
pub enum CppnNode<A: ActivationFunction> {
    Bias,
    Input,
    Output,
    Hidden(A),
}

const CPPN_BIAS_WEIGHT: f64 = 1.0;
impl<A: ActivationFunction> ActivationFunction for CppnNode<A> {
    fn formula(&self) -> &'static str {
        // XXX
        ""
    }

    fn calculate(&self, input: f64) -> f64 {
        match *self {
            CppnNode::Hidden(ref activation_function) => {
                // apply activation function on `input`
                activation_function.calculate(input)
            }
            CppnNode::Input | CppnNode::Output => {
                // Simply pass on the input signal.
                input
            }
            CppnNode::Bias => CPPN_BIAS_WEIGHT,
        }
    }
}

impl<A: ActivationFunction> NodeType for CppnNode<A> {
    fn accept_incoming_links(&self) -> bool {
        match *self {
            CppnNode::Input | CppnNode::Bias => false,
            _ => true,
        }
    }

    fn accept_outgoing_links(&self) -> bool {
        match *self {
            CppnNode::Output => false,
            _ => true,
        }
    }
}

impl<A: ActivationFunction> CppnNodeType for CppnNode<A> {
    fn is_input_node(&self) -> bool {
        match *self {
            CppnNode::Input => true,
            _ => false,
        }
    }
    fn is_output_node(&self) -> bool {
        match *self {
            CppnNode::Output => true,
            _ => false,
        }
    }
}

pub type CppnGraph<N: CppnNodeType> = Network<N, f64>;

/// Represents a Compositional Pattern Producing Network (CPPN)
pub struct Cppn<'a, N: CppnNodeType + 'a> {
    graph: &'a CppnGraph<N>,
    inputs: Vec<CppnNodeIndex>,
    outputs: Vec<CppnNodeIndex>,

    // For each node in `graph` there exists a corresponding field in `incoming_signals` describing
    // the sum of all input signals for that node.  We could store it inline in the `CppnNode`, but
    // this would require to make the whole CppnGraph mutable.
    incoming_signals: Vec<f64>,
}

impl<'a, N: CppnNodeType> Cppn<'a, N> {
    pub fn new(graph: &'a CppnGraph<N>) -> Cppn<'a, N> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        graph.each_node_with_index(|node, index| {
            if node.node_type().is_input_node() {
                inputs.push(index);
            }
            if node.node_type().is_output_node() {
                outputs.push(index);
            }
        });

        Cppn {
            graph: graph,
            inputs: inputs,
            outputs: outputs,
            incoming_signals: graph.nodes().iter().map(|_| 0.0).collect(),
        }
    }

    fn set_signal(&mut self, node_idx: CppnNodeIndex, value: f64) {
        self.incoming_signals[node_idx.index()] = value;
    }

    fn reset_signals(&mut self) {
        for value in self.incoming_signals.iter_mut() {
            *value = 0.0;
        }
    }

    /// Forward-propagate the signals starting from `from_nodes`. We use
    /// breadth-first-search (BFS).
    fn propagate_signals(&mut self, mut nodes: Vec<CppnNodeIndex>, mut seen: FixedBitSet) {
        while let Some(node_idx) = nodes.pop() {
            let input = self.incoming_signals[node_idx.index()];
            let output = self.graph.node(node_idx).node_type().calculate(input);

            // propagate output signal to outgoing links.
            self.graph.each_active_forward_link_of_node(node_idx, |out_node_idx, weight| {
                let out_node = out_node_idx.index();
                self.incoming_signals[out_node] += weight * output;
                if !seen.contains(out_node) {
                    seen.insert(out_node);
                    nodes.push(out_node_idx);
                }
            });
        }
    }

    /// Calculate all outputs
    pub fn calculate(&mut self, inputs: &[&[f64]]) -> Vec<f64> {
        assert!(self.incoming_signals.len() == self.graph.nodes().len());
        self.reset_signals();

        // assign all inputs
        let mut i = 0;
        for input_list in inputs.iter() {
            for &input in input_list.iter() {
                let input_idx = self.inputs[i];
                self.set_signal(input_idx, input);
                i += 1;
            }
        }
        assert!(i == self.inputs.len());

        let mut nodes = Vec::new(); // XXX: worst case capacity
        let mut seen = FixedBitSet::with_capacity(self.incoming_signals.len());

        // start from all nodes which have zero in_degree()
        self.graph.each_node_with_index(|node, index| {
            if node.in_degree() == 0 {
                nodes.push(index);
                seen.insert(index.index());
            }
        });

        // propagate the signals starting from the nodes with zero in degree.
        self.propagate_signals(nodes, seen);

        self.outputs
            .iter()
            .map(|&node_idx| self.incoming_signals[node_idx.index()])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use bipolar::BipolarActivationFunction as AF;
    use super::{Cppn, CppnGraph, CppnNode};
    use acyclic_network::ExternalNodeId;
    use rand;

    #[test]
    fn test_cycle() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::Input, ExternalNodeId(1));
        let h1 = g.add_node(CppnNode::Hidden(AF::Identity), ExternalNodeId(2));
        let h2 = g.add_node(CppnNode::Hidden(AF::Identity), ExternalNodeId(3));
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
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::Input, ExternalNodeId(1));
        let h1 = g.add_node(CppnNode::Hidden(AF::Linear), ExternalNodeId(2));
        let o1 = g.add_node(CppnNode::Output, ExternalNodeId(3));
        g.add_link(i1, h1, 0.5);
        g.add_link(h1, o1, 1.0);

        let mut cppn = Cppn::new(&g);

        assert_eq!(vec![0.5 * 0.5], cppn.calculate(&[&[0.5]]));
        assert_eq!(vec![1.0], cppn.calculate(&[&[4.0]]));
        assert_eq!(vec![-1.0], cppn.calculate(&[&[-4.0]]));
    }

    #[test]
    fn test_find_random_unconnected_link_no_cycle() {
        let mut g = CppnGraph::<CppnNode<AF>>::new();
        let i1 = g.add_node(CppnNode::Input, ExternalNodeId(1));
        let o1 = g.add_node(CppnNode::Output, ExternalNodeId(2));
        let o2 = g.add_node(CppnNode::Output, ExternalNodeId(3));

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

}
