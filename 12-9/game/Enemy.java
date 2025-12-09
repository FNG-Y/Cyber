package game;

public class Enemy {
    private String name;
    private int currentHP;
    private int maxHP;

    public Enemy(String name, int maxHP) {
        setName(name);
        setMaxHP(maxHP);
        this.currentHP = maxHP;
    }

    private void setMaxHP(int maxHP) {
        if (maxHP <= 0) {
            throw new IllegalArgumentException("Max HP must be positive");
        }
        this.maxHP = maxHP;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        if (name == null || name.trim().length() < 3) {
            throw new IllegalArgumentException("Name must be at least 3 characters");
        }
        this.name = name.trim();
    }

    public int getCurrentHP() {
        return currentHP;
    }

    public void setCurrentHP(int currentHP) {
        this.currentHP = Math.max(0, Math.min(currentHP, maxHP));
    }

    public int getMaxHP() {
        return maxHP;
    }

    public void attack(Hero hero) {
        hero.setCurrentHP(hero.getCurrentHP() - 2);
    }
}