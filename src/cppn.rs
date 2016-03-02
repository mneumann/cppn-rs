use activation_function::ActivationFunction;
use acyclic_network::{NodeType, Network};
pub use acyclic_network::NodeIndex as CppnNodeIndex;
use fixedbitset::FixedBitSet;
use std::fmt::Debug;

pub trait CppnNodeType: NodeType + ActivationFunction {
    fn is_input_node(&self) -> bool;
    fn is_output_node(&self) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub enum CppnNodeKind {
    Bias,
    Input,
    Output,
    Hidden,
}

/// A concrete implementation of a CppnNodeType.

#[derive(Clone, Debug)]
pub struct CppnNode<A: ActivationFunction> {
    kind: CppnNodeKind,
    activation_function: A,
}

impl<A> CppnNode<A> where A: ActivationFunction {
    pub fn new(kind: CppnNodeKind, activation_function: A) -> Self {
        CppnNode {
            kind: kind,
            activation_function: activation_function 
        }
    }

    pub fn input(activation_function: A) -> Self {
        Self::new(CppnNodeKind::Input, activation_function) 
    }

    pub fn output(activation_function: A) -> Self {
        Self::new(CppnNodeKind::Output, activation_function) 
    }

    pub fn hidden(activation_function: A) -> Self {
        Self::new(CppnNodeKind::Hidden, activation_function) 
    }

    pub fn bias(activation_function: A) -> Self {
        Self::new(CppnNodeKind::Bias, activation_function) 
    }

}

impl<A: ActivationFunction> ActivationFunction for CppnNode<A> {
    fn formula_gnuplot(&self, x: String) -> String {
        match self.kind {
            CppnNodeKind::Input | CppnNodeKind::Output | CppnNodeKind::Hidden | CppnNodeKind::Bias => {
                self.activation_function.formula_gnuplot(x)
            }
        }
    }

    fn calculate(&self, input: f64) -> f64 {
        self.activation_function.calculate(input)
    }
}

impl<A: ActivationFunction> NodeType for CppnNode<A> {
    fn accept_incoming_links(&self) -> bool {
        match self.kind {
            CppnNodeKind::Hidden | CppnNodeKind::Output => true,
            CppnNodeKind::Input | CppnNodeKind::Bias => false,
        }
    }

    fn accept_outgoing_links(&self) -> bool {
        match self.kind {
            CppnNodeKind::Hidden | CppnNodeKind::Input | CppnNodeKind::Bias => true,
            CppnNodeKind::Output => false,
        }
    }
}

impl<A: ActivationFunction> CppnNodeType for CppnNode<A> {
    fn is_input_node(&self) -> bool {
        match self.kind {
            CppnNodeKind::Input => true,
            _ => false,
        }
    }
    fn is_output_node(&self) -> bool {
        match self.kind {
            CppnNodeKind::Output => true,
            _ => false,
        }
    }
}

pub type CppnGraph<N, L, EXTID>
    where N: CppnNodeType,
          L: Copy + Debug + Send + Sized + Into<f64>,
          EXTID: Copy + Debug + Send + Sized + Ord = Network<N, L, EXTID>;

/// Represents a Compositional Pattern Producing Network (CPPN)
pub struct Cppn<'a, N, L, EXTID>
    where N: CppnNodeType + 'a,
          L: Copy + Debug + Send + Sized + Into<f64> + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    graph: &'a CppnGraph<N, L, EXTID>,
    inputs: Vec<CppnNodeIndex>,
    outputs: Vec<CppnNodeIndex>,

    // For each node in `graph` there exists a corresponding field in `incoming_signals` describing
    // the sum of all input signals for that node.  We could store it inline in the `CppnNode`, but
    // this would require to make the whole CppnGraph mutable.
    incoming_signals: Vec<f64>,
}

impl<'a, N, L, EXTID> Cppn<'a, N, L, EXTID>
    where N: CppnNodeType + 'a,
          L: Copy + Debug + Send + Sized + Into<f64> + 'a,
          EXTID: Copy + Debug + Send + Sized + Ord + 'a
{
    pub fn new(graph: &'a CppnGraph<N, L, EXTID>) -> Cppn<'a, N, L, EXTID> {
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
                let weight: f64 = weight.into();
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
    use activation_function::GeometricActivationFunction as AF;
    use super::{Cppn, CppnGraph, CppnNode};
    use acyclic_network::ExternalId;
    use rand;

    #[test]
    fn test_cycle() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let h1 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let h2 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(3));
        assert_eq!(true, g.valid_link(i1, i1).is_err());
        assert_eq!(true, g.valid_link(h1, h1).is_err());

        assert_eq!(true, g.valid_link(h1, i1).is_err());
        assert_eq!(Ok(()), g.valid_link(i1, h1));
        assert_eq!(Ok(()), g.valid_link(i1, h2));
        assert_eq!(Ok(()), g.valid_link(h1, h2));

        g.add_link(i1, h1, 0.0, ExternalId(1));
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(false, g.link_would_cycle(i1, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
        assert_eq!(true, g.link_would_cycle(i1, i1));
        assert_eq!(false, g.link_would_cycle(h1, h2));
        assert_eq!(false, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(h2, i1));

        g.add_link(h1, h2, 0.0, ExternalId(2));
        assert_eq!(true, g.link_would_cycle(h2, i1));
        assert_eq!(true, g.link_would_cycle(h1, i1));
        assert_eq!(true, g.link_would_cycle(h2, h1));
        assert_eq!(false, g.link_would_cycle(i1, h2));
    }

    #[test]
    fn test_simple_cppn() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let h1 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let o1 = g.add_node(CppnNode::output(AF::Linear), ExternalId(3));
        g.add_link(i1, h1, 0.5, ExternalId(1));
        g.add_link(h1, o1, 1.0, ExternalId(2));

        let mut cppn = Cppn::new(&g);

        let f = |x| 0.5 * x * 1.0;
        assert_eq!(vec![f(0.5)], cppn.calculate(&[&[0.5]]));
        assert_eq!(vec![f(4.0)], cppn.calculate(&[&[4.0]]));
        assert_eq!(vec![f(-4.0)], cppn.calculate(&[&[-4.0]]));
    }

    #[test]
    fn test_find_random_unconnected_link_no_cycle() {
        let mut g: CppnGraph<CppnNode<AF>, _, _> = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let o1 = g.add_node(CppnNode::output(AF::Linear), ExternalId(2));
        let o2 = g.add_node(CppnNode::output(AF::Linear), ExternalId(3));

        let mut rng = rand::thread_rng();

        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        let l = link.unwrap();
        assert!((i1, o1) == l || (i1, o2) == l);

        g.add_link(i1, o2, 0.0, ExternalId(1));
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(true, link.is_some());
        assert_eq!((i1, o1), link.unwrap());

        g.add_link(i1, o1, 0.0, ExternalId(2));
        let link = g.find_random_unconnected_link_no_cycle(&mut rng);
        assert_eq!(false, link.is_some());
    }

}
