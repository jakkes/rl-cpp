import asyncio
from typing import AsyncIterator

import numpy as np
import gym
from grpclib.server import Server

from rlbuf.remote_env import cart_pole


def state_to_proto(state: np.ndarray) -> cart_pole.State:
    return cart_pole.State(
        position=state[0],
        velocity=state[1],
        angle=state[2],
        angular_velocity=state[3]
    )


class Service(cart_pole.CartPoleServiceBase):

    async def env_stream(self, action_iterator: AsyncIterator[cart_pole.Action]) -> AsyncIterator[cart_pole.Observation]:
        env = gym.make("CartPole-v1")
        state, _ = env.reset()
        state = state_to_proto(state)

        yield cart_pole.Observation(0.0, False, state)

        async for action in action_iterator:
            state, reward, terminal, truncated, _ = env.step(action.action)
            terminal = terminal or truncated
            if terminal:
                state, _ = env.reset()
            
            state = state_to_proto(state)
            yield cart_pole.Observation(
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
