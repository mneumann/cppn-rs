use activation_function::ActivationFunction;
use acyclic_network::{NodeType, Network, Node};
pub use acyclic_network::NodeIndex as CppnNodeIndex;
use fixedbitset::FixedBitSet;

#[derive(Clone, Debug)]
pub enum CppnNodeType<A: ActivationFunction> {
    Bias,
    Input,
    Output,
    Hidden(A),
}

const CPPN_BIAS_WEIGHT: f64 = 1.0;

impl<A: ActivationFunction> NodeType for CppnNodeType<A> {
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

pub type CppnNode<A: ActivationFunction> = Node<CppnNodeType<A>, f64>;
pub type CppnGraph<A: ActivationFunction> = Network<CppnNodeType<A>, f64>;

/// Represents a Compositional Pattern Producing Network (CPPN)
pub struct Cppn<'a, A: ActivationFunction + 'a> {
    graph: &'a CppnGraph<A>,
    inputs: Vec<CppnNodeIndex>,
    outputs: Vec<CppnNodeIndex>,

    // For each node in `graph` there exists a corresponding field in `incoming_signals` describing
    // the sum of all input signals for that node.  We could store it inline in the `CppnNode`, but
    // this would require to make the whole CppnGraph mutable.
    incoming_signals: Vec<f64>,
}

impl<'a, A: ActivationFunction> Cppn<'a, A> {
    pub fn new(graph: &'a CppnGraph<A>) -> Cppn<'a, A> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for (i, node) in graph.nodes().iter().enumerate() {
            match *node.node_type() {
                CppnNodeType::Input => {
                    inputs.push(CppnNodeIndex::new(i));
                }
                CppnNodeType::Output => {
                    outputs.push(CppnNodeIndex::new(i));
                }
                _ => {}
            }
        }

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
            let node = &self.graph.nodes()[node_idx.index()];
            let input = self.incoming_signals[node_idx.index()];
            let output = match *node.node_type() {
                CppnNodeType::Hidden(ref activation_function) => {
                    // apply activation function on `input`
                    activation_function.calculate(input)
                }
                CppnNodeType::Input | CppnNodeType::Output => {
                    // Simply pass on the input signal.
                    input
                }
                CppnNodeType::Bias => CPPN_BIAS_WEIGHT,
            };

            // propagate output signal to outgoing links.
            node.each_active_forward_link(|out_node_idx, weight| {
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

        let mut nodes = Vec::new(); // XXX: worst case capacity
        let mut seen = FixedBitSet::with_capacity(self.incoming_signals.len());

        // assign all inputs
        let mut i = 0;
        for input_list in inputs.iter() {
            for &input in input_list.iter() {
                let input_idx = self.inputs[i];
                self.set_signal(input_idx, input);
                nodes.push(input_idx);
                seen.insert(input_idx.index());
                i += 1;
            }
        }
        assert!(i == self.inputs.len());

        // propagate the signals starting from the input signals.
        self.propagate_signals(nodes, seen);

        self.outputs
            .iter()
            .map(|&node_idx| self.incoming_signals[node_idx.index()])
            .collect()
    }
}


#[test]
fn test_cycle() {
    use super::bipolar::BipolarActivationFunction as AF;
    let mut g = CppnGraph::new();
    let i1 = g.add_node(CppnNodeType::Input);
    let h1 = g.add_node(CppnNodeType::Hidden(AF::Identity));
    let h2 = g.add_node(CppnNodeType::Hidden(AF::Identity));
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
    use super::bipolar::BipolarActivationFunction as AF;

    let mut g = CppnGraph::new();
    let i1 = g.add_node(CppnNodeType::Input);
    let h1 = g.add_node(CppnNodeType::Hidden(AF::Linear));
    let o1 = g.add_node(CppnNodeType::Output);
    g.add_link(i1, h1, 0.5);
    g.add_link(h1, o1, 1.0);

    let mut cppn = Cppn::new(&g);

    assert_eq!(vec![0.5 * 0.5], cppn.calculate(&[&[0.5]]));
    assert_eq!(vec![1.0], cppn.calculate(&[&[4.0]]));
    assert_eq!(vec![-1.0], cppn.calculate(&[&[-4.0]]));
}


#[test]
fn test_find_random_unconnected_link_no_cycle() {
    use rand;
    use super::bipolar::BipolarActivationFunction as AF;

    let mut g = CppnGraph::<AF>::new();
    let i1 = g.add_node(CppnNodeType::Input);
    let o1 = g.add_node(CppnNodeType::Output);
    let o2 = g.add_node(CppnNodeType::Output);

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
