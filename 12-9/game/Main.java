package game;

public class Main {
    public static void main(String[] args) {
        // 1. 初始化战斗场（对应你示例中的 service）
        BattleArena arena = new BattleArena();

        // 2. 创建英雄（名称：Arthur，最大HP：50）
        Hero hero = new Hero("Arthur", 50);
        // 创建敌人（名称：Goblin，最大HP：20）
        Enemy enemy = new Enemy("Goblin", 20);

        // 3. 创建图书馆和技能书
        Library library = new Library();
        Book swordBook = new Book("Sword Guide", "Master Lee", 100, Skill.SWORDPLAY);
        Book fireBook = new Book("Fire Magic", "Wizard Bob", 80, Skill.FIREBALL);
        library.addBook(swordBook);
        library.addBook(fireBook);

        // 4. 给英雄分配最优技能书（FIREBALL）
        System.out.println("Assign fire skill to hero: " + library.assignBestSkillBook(hero, Skill.FIREBALL));
        // 打印英雄当前技能
        System.out.println("Hero current skill: " + hero.getCurrentSkill());

        // 5. 英雄攻击敌人（测试攻击方法）
        hero.attack(enemy);
        System.out.println("Enemy HP after hero attack: " + enemy.getCurrentHP());

        // 6. 敌人反击英雄
        enemy.attack(hero);
        System.out.println("Hero HP after enemy attack: " + hero.getCurrentHP());

        // 7. 完整战斗（直到一方战败）
        System.out.println("Hero win the battle: " + BattleArena.fightToTheEnd(hero, enemy));

        // 8. 打印最终状态
        System.out.println("Final Hero HP: " + hero.getCurrentHP());
        System.out.println("Final Enemy HP: " + enemy.getCurrentHP());
    }
}