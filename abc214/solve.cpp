#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

using namespace std;

struct node {
    node* l = nullptr;
    node* r = nullptr;
    int x, y;
    int sz = 1;
    int mod = -1;
    int cnt[3] = {0, 0, 0};

    node(int x): x(x) {
        y = rand();
        cnt[x] = 1;
    }
};

int s(node* v) {
    if (!v) {
        return 0;
    }
    return v->sz;
}

int c(node* v, int i) {
    if (!v) {
        return 0;
    }
    return v->cnt[i];
}

void upd(node* v) {
    if (!v) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        v->cnt[i] = (v->x == i) + c(v->l, i) + c(v->r, i);
    }
    v->sz = 1 + s(v->l) + s(v->r);
}

void apply_tag(node* v, int x) {
    if (!v) return;
    v->x = x;
    for (int i = 0; i < 3; ++i) v->cnt[i] = 0;
    v->cnt[x] = v->sz;
    v->mod = x;
}

void push(node* v) {
    if (!v || v->mod == -1) return;
    apply_tag(v->l, v->mod);
    apply_tag(v->r, v->mod);
    v->mod = -1;
}

node* merge(node* l, node* r) {
    push(l);
    push(r);
    if (!l) return r;
    if (!r) return l;
    if (l->y > r->y) {
        l->r = merge(l->r, r);
        upd(l);
        return l;
    }
    r->l = merge(l, r->l);
    upd(r);
    return r;
}

typedef pair<node*, node*> nodes;

nodes split(node* p, int k) {
    push(p);
    if (!p) {
        return {nullptr, nullptr};
    }
    if (s(p->l) + 1 <= k) {
        nodes q = split(p->r, k - s(p->l) - 1);
        p->r = q.first;
        upd(p);
        return {p, q.second};
    }
    nodes q = split(p->l, k);
    p->l = q.second;
    upd(p);
    return {q.first, p};
}

node* root = nullptr;

void set_segment(node*& root, int l, int r, int x) {
    if (l > r) {
        return;
    }
    nodes q = split(root, r);
    nodes q1 = split(q.first, l - 1);
    apply_tag(q1.second, x);
    root = merge(q1.first, merge(q1.second, q.second));
}

void make_sort(int l, int r) {
    nodes q = split(root, r);
    nodes q1 = split(q.first, l - 1);
    int cnt[3] = {0, 0, 0};
    push(q1.second);
    for (int i = 0; i < 3; ++i) {
        cnt[i] = q1.second->cnt[i];
    }
    int b = 0;
    for (int i = 0; i < 3; ++i) {
        set_segment(q1.second, b + 1, b + cnt[i], i);
        b += cnt[i];
    }
    root = merge(q1.first, merge(q1.second, q.second));
}

void make_rsort(int l, int r) {
    nodes q = split(root, r);
    nodes q1 = split(q.first, l - 1);
    int cnt[3] = {0, 0, 0};
    push(q1.second);
    for (int i = 0; i < 3; ++i) {
        cnt[i] = q1.second->cnt[i];
    }
    int b = 0;
    for (int i = 2; i >= 0; --i) {
        set_segment(q1.second, b + 1, b + cnt[i], i);
        b += cnt[i];
    }
    root = merge(q1.first, merge(q1.second, q.second)); 
}

const int MAXN = 200'000;
int arr[MAXN];
int ptr = 0;

void print(node* v) {
    if (!v) {
        return;
    }
    push(v);
    print(v->l);
    arr[ptr++] = v->x;
    print(v->r);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int n, q, x;
    cin >> n >> q >> x;
    vector<int> p(n);
    for (int& i : p) {
        cin >> i;
    }
    for (int &i : p) {
        if (i < x) {
            i = 0;
        } else if (i == x) {
            i = 1;
        } else {
            i = 2;
        }
    }
    for (int i : p) {
        root = merge(root, new node(i));
    }
    assert(count(p.begin(), p.end(), 1) == 1);
    while (q--) {
        int c, l, r;
        cin >> c >> l >> r;
        if (c == 1) {
            make_sort(l, r);
        } else {
            make_rsort(l, r);
        }
    }
    print(root);
    for (int i = 0; i < n; ++i) {
        if (arr[i] == 1) {
            cout << i + 1 << endl;
        }
    }
}
