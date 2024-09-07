import numpy as np
import os 

def evalSpice(filename):
    # Check if the file exists and is not empty
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        raise FileNotFoundError('Please give the name of a valid SPICE file as input')

    # Read the file contents
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Initialize variables for parsing
    in_circuit = False
    components = []
    nodes = set()

    # Parse the file to extract circuit components between .circuit and .end
    for line in lines:
        line = line.strip()
        if line.startswith('.circuit'):
            in_circuit = True
            continue
        if line.startswith('.end'):
            break
        if in_circuit:
            if '#' in line:
                line = line.split('#')[0].strip()  # Remove comments
            if line:
                components.append(line.split())

    # Check if any components were found
    if not components:
        raise ValueError('Malformed circuit file')

    # Initialize node mapping and process components
    node_map = {'GND': 0}
    node_index = 1
    mod_components = [] # List to store components to be processed

    for comp in components:
        if comp[0][0] in {'R', 'V', 'I'}:
            mod_components.append(comp)
            nodes.update(comp[1:3]) # Add nodes to the set
        else:
            raise ValueError('Only V, I, R elements are permitted')

    # Map nodes to indices
    for node in nodes:
        if node not in node_map:
            node_map[node] = node_index
            node_index += 1

    # Calculate the size of the matrix
    num_nodes = len(node_map)
    num_vsources = sum(1 for comp in mod_components if comp[0].startswith('V'))
    M = num_nodes + num_vsources - 1

    # Initialize the conductance matrix G and current vector I
    G = np.zeros((M, M))
    I = np.zeros(M)

    # v_index tracks the additional rows/columns in G for voltage sources
    v_index = num_nodes - 1

    # Fill the matrices based on the components
    for comp in mod_components:
        comp_type = comp[0][0]
        n1, n2 = node_map[comp[1]], node_map[comp[2]]
        value = float(comp[-1])
        if comp_type == 'R':
            if n1 != 0: 
                G[n1-1, n1-1] += 1 / value  # Add conductance to diagonal elements
            if n2 != 0:
                G[n2-1, n2-1] += 1 / value
            if n1 != 0 and n2 != 0:
                G[n1-1, n2-1] -= 1 / value  # Subtract conductance from off-diagonal elements
                G[n2-1, n1-1] -= 1 / value

        elif comp_type == 'V':
            if n1 != 0:
                G[n1-1, v_index] = 1
                G[v_index, n1-1] = 1
            if n2 != 0:
                G[n2-1, v_index] = -1
                G[v_index, n2-1] = -1
            I[v_index] = value  # Set the corresponding entry in I to the source value
            v_index += 1

        elif comp_type == 'I':
            if n1 != 0:
                I[n1-1] -= value
            if n2 != 0:
                I[n2-1] += value

    # Solve the system of equations Gx = I
    try:
        x = np.linalg.solve(G, I)
    except np.linalg.LinAlgError:
        raise ValueError('Circuit error: no solution')

    # Extract node voltages
    V = {node: float(x[node_map[node]-1]) if node_map[node] != 0 else 0.0 for node in node_map}

    # Extract currents through voltage sources
    idx = 0
    I = {}
    for comp in mod_components:
        if comp[0][0] == 'V':
            I[comp[0]] = float(x[num_nodes + idx - 1])
            idx += 1
    
    # Return the node voltages and the currents through voltage sources
    return V, I
