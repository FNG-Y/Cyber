package game;

public class BattleArena {
    public static boolean fightToTheEnd(Hero hero, Enemy enemy) {
        while (true) {
            // 英雄攻击
            hero.attack(enemy);
            if (enemy.getCurrentHP() == 0) {
                return true; // 英雄胜利
            }

            // 敌人攻击
            enemy.attack(hero);
            if (hero.getCurrentHP() == 0) {
                return false; // 敌人胜利
            }
        }
    }
}