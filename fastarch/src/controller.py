from .types import Node
import time
import datetime

# nodes is a list of nodes
def run_system(state_machine, nodes, number_cycles, step=False):
	
	# collect connections
	connections = []
	for node in nodes:
		for port_name in node.ports:
			if node.ports[port_name].is_requester:
				connections.append(node.ports[port_name].connection)
	
	# printing outputs vars
	last_percent = 0
	
	# start time
	start_time = time.time()
	last_update_time = start_time
	
	# for each cycle
	current_cycle = 0
	while (not state_machine.finished()) and current_cycle <= number_cycles:
		# update state engine
		state_machine.update(current_cycle)
		# pre-step update
		for node in nodes:
			node.pre_step(current_cycle)
		# update requests
		for node in nodes:
			node.step_request(current_cycle)
		# update ready signals
		for node in nodes:
			node.step_ready(current_cycle)
		# pass data
		for connection in connections:
			connection.update(current_cycle)
		#for source in nodes:
		#	# look at each node as a source; data is pushed from source to sink
		#	for sink_name in source.sinks:
		#		sink = source.sinks[sink_name]
		#		if source.sinks_request[sink_name] == None:
		#			continue
		#		if source.sinks_ready[sink_name] > 0 and not (source.sinks_request[sink_name] == None):
		#			assert source.sinks_ready[sink_name] <= source.sinks_bitwidth[sink_name] # ready bits cannot be greater than bitwidth
		#			amount_to_pass = min(source.sinks_request[sink_name][0], source.sinks_ready[sink_name])
		#			source.send_data(sink, amount_to_pass)
		#			sink.receive_data(source, amount_to_pass)
		# post-step update
		for node in nodes:
			node.post_step(current_cycle)
		current_cycle += 1
		
		# update percent done every 10 cycles
		if current_cycle % 10 == 0:
			curr_time = time.time()
			if curr_time - last_update_time >= 5:
				last_update_time = curr_time
				print("{} - {:2.2%}".format(str(datetime.timedelta(seconds=int(curr_time - start_time))), state_machine.get_percent_complete()))
		
		if step:
			input()
	
	state_machine.on_complete(current_cycle)