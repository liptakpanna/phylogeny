from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
import numpy
import evalEnv
import log

def acc(topo, policy):
    acc_py_env = evalEnv.PhylogenyEvalEnv(topo)
    acc_env = tf_py_environment.TFPyEnvironment(acc_py_env)

    environment = acc_env
    total_return = 0.0
    max = 0.0
    db = 100
    for i in range(db):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        if episode_return > max: max = episode_return

        print(str(i) + ". ep return:" + str(episode_return))

    avg_return = total_return / db
    
    text = "Topo: "+topo+" result: "+ str(avg_return.numpy()[0])
    log.loggingAcc(text)

    if max == 0.0:
        print(avg_return.numpy()[0])
        print(max)
    else:
        print(avg_return.numpy()[0])
        print(max.numpy())