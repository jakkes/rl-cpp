import asyncio
from typing import AsyncIterator

import numpy as np
import gym
from grpclib.server import Server

from rlbuf.remote_env import lunar_lander


def state_to_proto(state: np.ndarray) -> lunar_lander.State:
    return lunar_lander.State(data=state.flatten().tolist())


class Service(lunar_lander.LunarLanderServiceBase):

    async def env_stream(self, action_iterator: AsyncIterator[lunar_lander.Action]) -> AsyncIterator[lunar_lander.Observation]:
        env = gym.make("LunarLander-v2")
        state, _ = env.reset()
        state = state_to_proto(state)

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


async def main():
    server = Server([Service()])
    await server.start("localhost", 50051)
    await server.wait_closed()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
