import asyncio
import multiprocessing
import time
import grpclib
from typing import AsyncIterator

import numpy as np
import gym
from grpclib.server import Server

from rlbuf.remote_env import lunar_lander


WORKERS = 16
BASE_PORT = 50500


def state_to_proto(state: np.ndarray) -> lunar_lander.State:
    return lunar_lander.State(data=state.flatten().tolist())


class Service(lunar_lander.LunarLanderServiceBase):

    async def env_stream(self, action_iterator: AsyncIterator[lunar_lander.Action]) -> AsyncIterator[lunar_lander.Observation]:
        env = gym.make("LunarLander-v2")
        state, _ = env.reset()
        state = state_to_proto(state)

        print("New connection")

        yield lunar_lander.Observation(0.0, False, state)

        async for action in action_iterator:
            state, reward, terminal, truncated, _ = env.step(action.action)
            terminal = terminal or truncated
            if terminal:
                state, _ = env.reset()
            
            state = state_to_proto(state)
            yield lunar_lander.Observation(
                reward=reward,
                terminal=terminal,
                next_state=state
            )


class Worker(multiprocessing.Process):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self._port = port

    async def _run(self):
        server = grpclib.server.Server([Service()])
        await server.start("localhost", self._port)
        await server.wait_closed()

    def run(self) -> None:
        super().run()
        asyncio.run(self._run())


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    workers = [ Worker(BASE_PORT + i) for i in range(WORKERS) ]
    for worker in workers:
        worker.start()

    while True:
        time.sleep(1.0)

