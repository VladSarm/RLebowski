I) "Наша задача сейчас начать реализовывать поверх REINFORCE PPO. Я тебе опишу мой план того, как это сделать, сначала ты ответь на вопрос действительно ли это является PPO алгоритмом, затем может приступить к выполнению
И так, основным изменениям у нас подвергнется изменение вычисления loss функции здесь
def _compute_returns_and_update(log_probs_per, rewards_per):
    """Compute discounted returns, normalize, do one gradient update."""
    batch_log_probs = []
    batch_returns   = []
    ep_totals       = []

    for log_probs, rewards in zip(log_probs_per, rewards_per):
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
        ep_totals.append(sum(rewards))

    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    log_probs_t = torch.stack(batch_log_probs)
    loss = -torch.sum(log_probs_t * returns_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
Новый лосс считается так exp(log_probs_old - log_probs_new)*adavtage -> min(r*A, clip(r, 1-epsilon,1+epsilon)*A)
Для каждого rollout batch мы несколько раз вычисляем этот лос и делаем backprop. На первой итерации log_probs_old = log_probs_new
Наша новая политика совпадает со старой и лосс буквально становится тем же самы как в Reinforce  просто Advantage (но с добавлением клипа)
Затем в цикли n число раз (это гиперпараметр) мы считаем для ТЕХ ЖЕ observstions and sampled actions новые log_probs (то есть нужно по всеми батчу начинать хранить наблюдения на каждом шаге и выполненные действия), мы делаем для политики после backward новый sample_log_probs, считаем заново новый loss и делаем backward
Допущения на данном этапе 1) наш Advantage это (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8) , пока будем вычислять так, но имей ввиду что потом мы применим GAE для его вычисления
2) Мы делаем фиксированное число шагов обновления весов, а не смотрим какой процент уперся в clip и не отсекаем например по порогу 80%
3) Мы берем всю траекторию, а не отдельные случайные шаги из неё

Нужно будет добавить хранение нужной информации"


II) 