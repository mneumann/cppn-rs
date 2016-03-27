use activation_function::ActivationFunction;
use acyclic_network::{NodeType, Network};
pub use acyclic_network::NodeIndex as CppnNodeIndex;
use fixedbitset::FixedBitSet;
use std::fmt::Debug;

pub trait CppnNodeType: NodeType + ActivationFunction {
    fn is_input_node(&self) -> bool;
    fn is_output_node(&self) -> bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CppnNodeKind {
    Bias,
    Input,
    Output,
    Hidden,
}

/// A concrete implementation of a CppnNodeType.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CppnNode<A: ActivationFunction> {
    pub kind: CppnNodeKind,
    pub activation_function: A,
}

impl<A> CppnNode<A>
    where A: ActivationFunction
{
    pub fn new(kind: CppnNodeKind, activation_function: A) -> Self {
        CppnNode {
            kind: kind,
            activation_function: activation_function,
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
            CppnNodeKind::Input |
            CppnNodeKind::Output |
            CppnNodeKind::Hidden |
            CppnNodeKind::Bias => self.activation_function.formula_gnuplot(x),
        }
    }

    fn name(&self) -> String {
        self.activation_function.name()
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
    start_nodes: Vec<CppnNodeIndex>,

    // nodes array and bitarray used in BFS
    nodes_bfs: Vec<CppnNodeIndex>,
    seen_bfs: FixedBitSet,

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
        let mut start_nodes = Vec::new();

        graph.each_node_with_index(|node, index| {
            if node.node_type().is_input_node() {
                inputs.push(index);
            }
            if node.node_type().is_output_node() {
                outputs.push(index);
            }
            if node.in_degree() == 0 {
                start_nodes.push(index);
            }
        });

        let incoming_signals: Vec<_> = graph.nodes().iter().map(|_| 0.0).collect();
        let seen_bfs = FixedBitSet::with_capacity(incoming_signals.len());

        Cppn {
            graph: graph,
            inputs: inputs,
            outputs: outputs,
            start_nodes: start_nodes,
            nodes_bfs: Vec::new(),
            seen_bfs: seen_bfs,
            incoming_signals: incoming_signals,
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

    pub fn incoming_signals(&self) -> &[f64] {
        &self.incoming_signals
    }

    /// Forward-propagate the signals starting from `from_nodes`. We use
    /// breadth-first-search (BFS).
    fn propagate_signals(&mut self) {
        while let Some(node_idx) = self.nodes_bfs.pop() {
            let input = self.incoming_signals[node_idx.index()];
            let output = self.graph.node(node_idx).node_type().calculate(input);

            // propagate output signal to outgoing links.
            self.graph.each_active_forward_link_of_node(node_idx, |out_node_idx, weight| {
                let out_node = out_node_idx.index();
                let weight: f64 = weight.into();
                self.incoming_signals[out_node] += weight * output;
                if !self.seen_bfs.contains(out_node) {
                    self.seen_bfs.insert(out_node);
                    self.nodes_bfs.push(out_node_idx);
                }
            });
        }
    }

    /// Calculate all outputs

    pub fn calculate(&mut self, inputs: &[&[f64]]) -> Vec<f64> {
        self.process(inputs);
        (0..self.outputs.len()).into_iter().map(|i| self.read_output(i).unwrap()).collect()
    }

    /// Reads the `nth_output` of the network.

    pub fn read_output(&self, nth_output: usize) -> Option<f64> {
        self.outputs.get(nth_output).map(|&node_idx| {
            let input = self.incoming_signals[node_idx.index()];
            let output = self.graph.node(node_idx).node_type().calculate(input);
            output
        })
    }

    /// Returns the number of outputs

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Returns the number of inputs

    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Process the network for the given `inputs`. Outputs can be read after this call using
    /// `read_output`.

    pub fn process(&mut self, inputs: &[&[f64]]) {
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

        self.nodes_bfs.clear();
        self.seen_bfs.clear();

        // start from all nodes which have zero in_degree()
        for &start_node_index in &self.start_nodes {
            self.nodes_bfs.push(start_node_index);
            self.seen_bfs.insert(start_node_index.index());
        }

        // propagate the signals starting from the nodes with zero in degree.
        self.propagate_signals();
    }

    /// Group the nodes into layers.
    pub fn group_layers(&self) -> Vec<Vec<usize>> {
        let ranks = self.layout();
        let mut pairs: Vec<(usize, usize)> = ranks.iter().enumerate().map(|(nodei, &rank)| (rank, nodei)).collect();
        pairs.sort_by_key(|p| p.0);
        pairs.reverse();

        let mut layers = Vec::new();

        let (mut current_rank, first_node) = pairs.pop().unwrap();
        let mut layer = vec![first_node];

        while let Some((rank, nodei)) = pairs.pop() {
            assert!(rank >= current_rank);
            if rank == current_rank {
                layer.push(nodei);
            } else {
                assert!(layer.len() > 0);
                layers.push(layer);
                layer = vec![nodei];
                current_rank = rank;
            }
        }
        assert!(layer.len() > 0);
        layers.push(layer);

        for layer in layers.iter_mut() {
            layer.sort();
        }

        layers
    }

    pub fn layout(&self) -> Vec<usize> {
        // each node has a rank (layer). All start initially 0 (same layer)
        let max_rank = self.graph.nodes().len() + 1; 
        let mut ranks: Vec<usize> = self.graph
                                       .nodes()
                                       .iter()
                                       .map(|node| {
                                           if node.node_type().is_input_node() {
                                               0
                                           } else if node.node_type().is_output_node() {
                                               max_rank
                                           } else {
                                               1
                                           }
                                       })
                                       .collect();

        loop {
            let mut changed = false;

            self.graph.each_node_with_index(|_node, index| {
                // make sure that the rank of all dependent links of a node are > the nodes rank
                self.graph.each_active_forward_link_of_node(index, |out_node_idx, _weight| {
                    let src_rank = ranks[index.index()];
                    let dst_rank = ranks[out_node_idx.index()];
                    if dst_rank <= src_rank {
                        ranks[out_node_idx.index()] = src_rank + 1;
                        changed = true;
                    }
                });
            });

            if !changed {
                break;
            }
        }
        ranks
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
    fn test_cppn_with_output_activation_function() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let h1 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let o1 = g.add_node(CppnNode::output(AF::Constant1), ExternalId(3));
        g.add_link(i1, h1, 0.5, ExternalId(1));
        g.add_link(h1, o1, 1.0, ExternalId(2));

        let mut cppn = Cppn::new(&g);

        assert_eq!(vec![1.0], cppn.calculate(&[&[0.5]]));
        assert_eq!(vec![1.0], cppn.calculate(&[&[4.0]]));
        assert_eq!(vec![1.0], cppn.calculate(&[&[-4.0]]));
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

    #[test]
    fn test_layout() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let h1 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let h2 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let o1 = g.add_node(CppnNode::output(AF::Constant1), ExternalId(3));
        g.add_link(i1, h1, 0.5, ExternalId(1));
        g.add_link(h1, o1, 1.0, ExternalId(2));

        assert_eq!(vec![0,1,1,5], Cppn::new(&g).layout());

        g.add_link(i1, h2, 0.5, ExternalId(1));
        assert_eq!(vec![0,1,1,5], Cppn::new(&g).layout());
        g.add_link(h2, o1, 0.5, ExternalId(1));
        assert_eq!(vec![0,1,1,5], Cppn::new(&g).layout());
        g.add_link(h2, h1, 0.5, ExternalId(1));
        assert_eq!(vec![0,2,1,5], Cppn::new(&g).layout());
    }

    #[test]
    fn test_group_layers() {
        let mut g = CppnGraph::new();
        let i1 = g.add_node(CppnNode::input(AF::Linear), ExternalId(1));
        let h1 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let h2 = g.add_node(CppnNode::hidden(AF::Linear), ExternalId(2));
        let o1 = g.add_node(CppnNode::output(AF::Constant1), ExternalId(3));
        g.add_link(i1, h1, 0.5, ExternalId(1));
        g.add_link(h1, o1, 1.0, ExternalId(2));

        assert_eq!(vec![vec![0], vec![1,2], vec![3]], Cppn::new(&g).group_layers());

        g.add_link(i1, h2, 0.5, ExternalId(1));
        assert_eq!(vec![vec![0], vec![1,2], vec![3]], Cppn::new(&g).group_layers());
        g.add_link(h2, o1, 0.5, ExternalId(1));
        assert_eq!(vec![vec![0], vec![1,2], vec![3]], Cppn::new(&g).group_layers());
        g.add_link(h2, h1, 0.5, ExternalId(1));
        assert_eq!(vec![vec![0], vec![2], vec![1], vec![3]], Cppn::new(&g).group_layers());
    }
}
