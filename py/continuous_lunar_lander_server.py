import asyncio
from typing import AsyncIterator

import numpy as np
import gym
from grpclib.server import Server

from rlbuf.env.remote import continuous_lunar_lander


def state_to_proto(state: np.ndarray) -> continuous_lunar_lander.State:
    return continuous_lunar_lander.State(data=state.flatten().tolist())


class Service(continuous_lunar_lander.ContinuousLunarLanderServiceBase):

    async def env_stream(self, action_iterator: AsyncIterator[continuous_lunar_lander.Action]) -> AsyncIterator[continuous_lunar_lander.Observation]:
        env = gym.make("LunarLander-v2", continuous=True)
        state, _ = env.reset()
        state = state_to_proto(state)

        yield continuous_lunar_lander.Observation(0.0, False, state)

        async for action in action_iterator:
            a = (action.main_engine, action.lateral_engine)
            state, reward, terminal, truncated, _ = env.step(a)
            terminal = terminal or truncated
            if terminal:
                state, _ = env.reset()
            
            state = state_to_proto(state)
            yield continuous_lunar_lander.Observation(
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
