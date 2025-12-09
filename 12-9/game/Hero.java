package game;

public class Hero {
    private String name;
    private int currentHP;
    private int maxHP;
    private Skill currentSkill;

    public Hero(String name, int maxHP) {
        setName(name);
        setMaxHP(maxHP);
        this.currentHP = maxHP;
        this.currentSkill = Skill.NONE;
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

    public Skill getCurrentSkill() {
        return currentSkill;
    }

    public void setCurrentSkill(Skill currentSkill) {
        if (currentSkill == null) {
            throw new IllegalArgumentException("Skill cannot be null");
        }
        this.currentSkill = currentSkill;
    }

    public void readBook(Book book) {
        this.currentSkill = book.getSkill();
        System.out.printf("%s has read '%s' and now knows: %s%n",
                name, book.getTitle(), currentSkill);
    }

    public void attack(Enemy enemy) {
        int damage = switch (currentSkill) {
            case SWORDPLAY -> 3;
            case FIREBALL -> 4;
            default -> 1;
        };
        enemy.setCurrentHP(enemy.getCurrentHP() - damage);
    }
}