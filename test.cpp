#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
#define SINGLE_TEST "YES"

const int N = 10;
ll n;
ll d[N][2];
vector<int> v;

vector<int> convert_vector(ll n) {
    vector<int> rs;
    while (n) {
        rs.push_back(n % 10);
        n /= 10;
    }
    reverse(rs.begin(), rs.end());
    return rs;
}

ll dp(int i = 0, int j = 1, ll sum = 0) {
    if (i >= v.size()) {
        return sum;
    }
    ll &rs = d[i][j];
    if (rs != -1) {
        return rs;
    }
    rs = 0;
    int mx = j ? v[i] : 9;

    for (int k = (mx - (mx % 2 == 0)); k >= 1; k -= 2) {
        cerr << "newj : " << i << ' ' << j << ' ' << k << " : " << (j & (k == mx)) << '\n';
        rs = max(rs, dp(i + 1, j & (k == mx), sum * 10 + k));
    }
    return rs;
}

ll nine(int siz) {
    ll rs = 0;
    for (int i = 1; i <= siz; ++i) {
        rs = rs * 10 + 9;
    }
    return rs;
}

void solve() {
    cin >> n;
    v = convert_vector(n - 1);
    memset(d, -1, sizeof d);
    cout << max(nine(v.size() - 1), dp());
}

signed main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int test = 1;
    if (SINGLE_TEST == "NO") {
        cin >> test;
    }
    while (test--) {
        solve();
    }
    return 0;
}